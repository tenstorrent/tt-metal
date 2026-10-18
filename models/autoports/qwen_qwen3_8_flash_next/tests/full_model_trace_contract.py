"""Repeated replay, sampler equivalence, cache, and request lifetime evidence."""

import argparse, json, time
from pathlib import Path
import torch, ttnn
from unittest.mock import patch
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def guard_runtime(fn):
    def forbidden(*args, **kwargs):
        raise AssertionError("Model/sampling runtime attempted a host tensor conversion")

    def guarded(*args, **kwargs):
        with (
            patch.object(ttnn, "from_torch", forbidden),
            patch.object(ttnn, "to_torch", forbidden),
            patch.object(torch, "tensor", forbidden),
            patch.object(torch, "as_tensor", forbidden),
        ):
            return fn(*args, **kwargs)

    return guarded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,3")
    ap.add_argument("--lengths", default="1,31,32,33,127,128,129,255,256,257,511,512,513,2053")
    ap.add_argument("--tag", default="trace_contract")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--stage", default="full_model")
    ap.add_argument("--prefill-trace-max", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    start = time.perf_counter()
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=[int(x) for x in args.layers.split(",") if x],
            batch_size=args.batch,
            max_seq_len=args.context,
            checkpoint=args.checkpoint,
            prefill_trace_max=args.prefill_trace_max,
        )
        gen.model.forward = guard_runtime(gen.model.forward)
        gen.sampler.decode_forward = guard_runtime(gen.sampler.decode_forward)
        gen.prepare()
        ids = dict(gen.traces)
        rows = []
        for n in map(int, args.lengths.split(",")):
            prompt = [100 + i % 300 for i in range(n)]
            a = gen.generate(prompt, 4)
            before = dict(gen.counters)
            b = gen.generate(prompt, 4)
            assert a == b, (n, a, b)
            assert ids == gen.traces
            assert gen.counters["captures"] == len(ids) and gen.counters["releases"] == 0
            rows.append(dict(length=n, tokens=a, repeat_exact=True, perf=gen.last_perf))
        # Inspect the actual persistent model token and position inputs. No
        # host writes occur between these two decode+sampling replays.
        gen.prefill_forward(
            torch.tensor([[100, 101, 102]]), page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[3]
        )
        before = dict(gen.counters)
        initial_token = host(gen.tt_token).reshape(-1).tolist()
        initial_pos = host(gen.tt_pos)[:, 0].tolist()
        steps = []
        for step in range(2):
            consumed = host(gen.tt_token).reshape(-1).tolist()
            positions = host(gen.tt_pos)[:, 0].tolist()
            gen._replay("decode")
            gen._replay("greedy")
            feedback = host(gen.tt_token).reshape(-1).tolist()
            nextpos = host(gen.tt_pos)[:, 0].tolist()
            assert nextpos == [p + 1 for p in positions]
            if steps:
                assert consumed == steps[-1]["feedback"]
            steps.append(dict(consumed=consumed, position=positions, feedback=feedback, next_position=nextpos))
        delta = {k: gen.counters[k] - before.get(k, 0) for k in gen.counters}
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "synchronizations"):
            assert delta.get(key, 0) == 0, (key, delta)
        # New page mapping at a request boundary, then unchanged mapping.
        newpage = gen.page_table.flip(1).contiguous()
        before = gen.counters["page_table_refreshes"]
        gen.prefill_forward(torch.tensor([[100, 101, 102]]), page_table=newpage, kv_cache=gen.kv_cache, prompt_lens=[3])
        assert gen.counters["page_table_refreshes"] == before + 1
        gen.decode_forward(None, None, page_table=newpage, kv_cache=gen.kv_cache)
        assert gen.counters["page_table_refreshes"] == before + 1
        # Both candidates are semantically greedy; physical32 candidates/shard.
        logits = host(gen.tt_logits)  # preserve rank0 only for metadata, not feedback
        for mode in ("greedy", "argmax", "sample", "greedy"):
            gen._replay(mode)
            if mode == "greedy":
                sampled = host(gen.tt_token).reshape(-1).tolist()
                if "expected" in locals():
                    assert sampled == expected
                expected = sampled
            elif mode == "argmax":
                assert host(gen.tt_token).reshape(-1).tolist() == expected
        assert ids == gen.traces
        report = dict(
            args=vars(args),
            rows=rows,
            feedback_steps=steps,
            steady_state_counter_delta=delta,
            changed_page_refreshes=1,
            trace_ids={k: str(v) for k, v in ids.items()},
            counters=dict(gen.counters),
            sampler_greedy_argmax_equal=True,
            dynamic_host_conversion_guards=True,
            elapsed_seconds=time.perf_counter() - start,
        )
        (root / "doc" / args.stage / f"{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TRACE_CONTRACT_PASS", json.dumps(report), flush=True)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
