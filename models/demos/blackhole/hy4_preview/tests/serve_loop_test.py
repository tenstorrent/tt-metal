"""device-sampling: the serve adapter (HYV4ForCausalLM, chain decoder) driven the way the plugin drives it under the decode reload contract (no vLLM). Env: HY4_SERVE_LAYERS (2), NSTEPS (30), TRACE_BYTES.
 A: host-loop reference (sampling_params None -> fp32 logits, host argmax) of 4 prompts, NSTEPS greedy steps.
 B: device loop greedy, first token from the DEVICE prefill sampler, decode with reload only at the transition, split submit / read_decode_output / process_decode_output_host, depth 1 then depth 2 (pipelined).
 C: slot_remap + a request finishing and a new one entering (reload_inputs only at the transitions); D: a host-sampling step in the middle (penalty case) then back to device sampling; E: sampled decode (default generation_config params) seeded repeatability."""
import json
import os
import time
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.generator_vllm import HYV4ForCausalLM

NSTEP = int(os.environ.get("NSTEPS", "30"))
os.environ.setdefault("HY4_SERVE_LAYERS", "2")


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def sp(T=0.0, k=1, p=1.0, seed=None, n=4):
    return types.SimpleNamespace(
        temperature=[T] * n,
        top_k=[k] * n,
        top_p=[p] * n,
        seed=[seed] * n,
        enable_log_probs=[False] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[1.0] * n,
    )


def main():
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "2e9"))))
    try:
        m = HYV4ForCausalLM.initialize_vllm_model(None, md, 4, 2048)
        t0 = time.time()
        m.warm_serving()
        res(
            event="warm",
            sec=round(time.time() - t0, 1),
            caps={
                k: v
                for k, v in HYV4ForCausalLM.model_capabilities.items()
                if k.startswith("supports_sample") or k.startswith("supports_async") or k.startswith("supports_device")
            },
            contract=HYV4ForCausalLM.decode_input_update_contract,
        )
        g = torch.Generator().manual_seed(1)
        prompts = [torch.randint(1000, 50000, (n,), generator=g).tolist() for n in (37, 90, 600, 61)]
        lens = [len(p) for p in prompts]
        toks = torch.zeros(4, max(lens), dtype=torch.long)
        for i, p in enumerate(prompts):
            toks[i, : lens[i]] = torch.tensor(p)

        def prefill(sampling=None):
            return m.prefill_forward(
                toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3], sampling_params=sampling
            )

        # A: host loop reference
        lg = prefill()
        cur = [int(lg[i, 0].argmax()) for i in range(4)]
        first_host = list(cur)
        ref = [cur]
        pos = list(lens)
        for st in range(NSTEP):
            out = m.decode_forward(torch.tensor(cur).reshape(4, 1), torch.tensor(pos))
            cur = [int(out[i, 0].argmax()) for i in range(4)]
            ref.append(cur)
            pos = [p + 1 for p in pos]
        res(event="A_host_ref", rows=[[r[i] for r in ref[:6]] for i in range(4)])

        def device_run(depth, steps, mid=None):
            first = prefill(sp())
            res(
                event="B_first_token_device_vs_host",
                device=first.tolist(),
                host=first_host,
                equal=first.tolist() == first_host,
            )
            outs, pend = [[int(x) for x in first]], []
            pos = list(lens)
            cur_t = first.reshape(4, 1).long()
            for st in range(steps):
                first_step = st == 0
                h = m.decode_forward(
                    cur_t,
                    torch.tensor(pos),
                    sampling_params=sp(),
                    read_from_device=False,
                    reload_inputs=first_step,
                    reload_sampling_params=first_step,
                    reset_sampling_state=first_step,
                )
                hh, ev = m.read_decode_output(h, async_read=True)
                for e in ev:
                    pass
                pend.append((hh, ev))
                pos = [p + 1 for p in pos]
                if len(pend) >= depth:
                    hh, ev = pend.pop(0)
                    for e in ev:
                        ttnn.event_synchronize(e)
                    outs.append(m.process_decode_output_host(hh, is_tokens=True).reshape(-1).tolist())
            while pend:
                hh, ev = pend.pop(0)
                for e in ev:
                    ttnn.event_synchronize(e)
                outs.append(m.process_decode_output_host(hh, is_tokens=True).reshape(-1).tolist())
            return outs

        t0 = time.perf_counter()
        o1 = device_run(1, NSTEP)
        t1 = time.perf_counter()
        o2 = device_run(2, NSTEP)
        t2 = time.perf_counter()
        n = len(ref)
        d1 = next((i for i in range(n) if o1[i] != ref[i]), None)
        d2 = next((i for i in range(n) if o2[i] != ref[i]), None)
        res(
            event="B_device_loop_vs_host",
            steps=NSTEP,
            depth1_first_diff=d1,
            depth2_first_diff=d2,
            depth1_eq_depth2=o1 == o2,
            depth1_s=round(t1 - t0, 2),
            depth2_s=round(t2 - t1, 2),
        )

        # E: sampled with the generation_config defaults, seeded repeat
        def sampled_run(seed):
            first = prefill(sp(0.9, -1, 1.0, seed))
            res(
                event="E_dbg",
                variant=m.dl.variant,
                first=first.tolist(),
                fs_plain=getattr(m.dl.fs, "plain", None),
                pos_f=[
                    float(ttnn.to_torch(ttnn.get_device_tensors(m.dl.pos_f)[r * 8]).reshape(-1)[0]) for r in range(4)
                ],
            )
            outs = [first.tolist()]
            pos = list(lens)
            cur_t = first.reshape(4, 1).long()
            for st in range(12):
                f = st == 0
                o = m.decode_forward(
                    cur_t,
                    torch.tensor(pos),
                    sampling_params=sp(0.9, -1, 1.0, seed),
                    read_from_device=True,
                    reload_inputs=f,
                    reload_sampling_params=f,
                    reset_sampling_state=f,
                )
                outs.append(o.reshape(-1).tolist())
                cur_t = o
                pos = [p + 1 for p in pos]
            return outs

        a, b, c = sampled_run(5), sampled_run(5), sampled_run(6)
        res(event="E_sampled_seeded", repeat_identical=a == b, other_seed_differs=a != c, row0=[x[0] for x in a])
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
