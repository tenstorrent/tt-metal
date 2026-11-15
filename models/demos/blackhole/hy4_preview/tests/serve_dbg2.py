import json
import os
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.generator_vllm import HYV4ForCausalLM

os.environ.setdefault("HY4_SERVE_LAYERS", "2")
res = lambda **kw: print("RES " + json.dumps(kw, default=str), flush=True)
sp = lambda T, k, p, seed=None, n=4: types.SimpleNamespace(
    temperature=[T] * n, top_k=[k] * n, top_p=[p] * n, seed=[seed] * n, enable_log_probs=[False] * n
)
md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "2e9"))))
try:
    m = HYV4ForCausalLM.initialize_vllm_model(None, md, 4, 2048)
    m.warm_serving()
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1000, 50000, (n,), generator=g).tolist() for n in (37, 90, 600, 61)]
    lens = [len(p) for p in prompts]
    toks = torch.zeros(4, max(lens), dtype=torch.long)
    for i, p in enumerate(prompts):
        toks[i, : lens[i]] = torch.tensor(p)
    probe = lambda tag: res(
        event="probe",
        after=tag,
        first=m.prefill_forward(
            toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3], sampling_params=sp(0.9, -1, 1.0, 5)
        ).tolist(),
    )
    probe("warm")
    lg = m.prefill_forward(toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3])
    cur = [int(lg[i, 0].argmax()) for i in range(4)]
    pos = list(lens)
    for st in range(5):
        out = m.decode_forward(torch.tensor(cur).reshape(4, 1), torch.tensor(pos))
        cur = [int(out[i, 0].argmax()) for i in range(4)]
        pos = [x + 1 for x in pos]
    probe("host_steps")
    first = m.prefill_forward(
        toks, start_pos=[0] * 4, prompt_lens=lens, empty_slots=[0, 1, 2, 3], sampling_params=sp(0.0, 1, 1.0)
    )
    cur_t = first.reshape(4, 1).long()
    pos = list(lens)
    for st in range(5):
        f = st == 0
        h = m.decode_forward(
            cur_t,
            torch.tensor(pos),
            sampling_params=sp(0.0, 1, 1.0),
            read_from_device=False,
            reload_inputs=f,
            reload_sampling_params=f,
            reset_sampling_state=f,
        )
        hh, ev = m.read_decode_output(h, async_read=True)
        for e in ev:
            ttnn.event_synchronize(e)
        o = m.process_decode_output_host(hh, is_tokens=True)
        pos = [x + 1 for x in pos]
    probe("device_split_steps")
finally:
    close_mesh(md)
