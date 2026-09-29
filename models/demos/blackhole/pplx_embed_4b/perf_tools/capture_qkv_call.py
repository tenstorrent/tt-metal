# Print the model's actual QKV projection call arguments (operand shapes, dtypes, memory configs incl. shard specs,
# program config, env-selected knobs) at a batch's perf config, from one eager prefill. The fused heads op is imported
# by name in tt/attention.py, so it is not hooked here. CAP_N=9728,19456 captures other projections by weight width
# (FF1 / FF3 9728, fused SwiGLU w13 19456, FF2 / WO 2560); default 6144 (QKV).
# Usage: capture_qkv_call.py [batch]
import os
import sys

import ttnn
from models.demos.blackhole.pplx_embed_4b.demo._common import (
    apply_workload_env,
    build_single_device_model,
    generate_synthetic_inputs,
)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CAP_N = {int(v) for v in os.getenv("CAP_N", "6144").split(",")}


def desc(t):
    if isinstance(t, ttnn.Tensor):
        mc = t.memory_config()
        return f"Tensor{tuple(t.shape)} {t.dtype} {mc!r}"
    return repr(t)[:300]


def _kw(v):
    if isinstance(v, ttnn.WormholeComputeKernelConfig):
        return (
            f"ckc(fidelity={v.math_fidelity}, approx={v.math_approx_mode}, fp32_acc={v.fp32_dest_acc_en}, "
            f"l1_acc={v.packer_l1_acc})"
        )
    return repr(v)


def main():
    apply_workload_env(B, 512)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000, num_command_queues=1)
    seen = {"mm": 0}
    orig_mm, orig_lin = ttnn.experimental.minimal_matmul, ttnn.linear

    def mm(a, b, *args, **kw):
        out = orig_mm(a, b, *args, **kw)
        if seen["mm"] < 8 and tuple(b.shape)[-1] in CAP_N:
            seen["mm"] += 1
            print(f"CAP mm in0 {desc(a)}\nCAP     in1 {desc(b)}\nCAP     out {desc(out)}", flush=True)
            for k, v in kw.items():
                print(f"CAP     {k} = {_kw(v)}"[:700], flush=True)
        return out

    def lin(a, b, *args, **kw):
        out = orig_lin(a, b, *args, **kw)
        if seen["mm"] < 8 and tuple(b.shape)[-1] in CAP_N:
            seen["mm"] += 1
            print(f"CAP linear in0 {desc(a)}\nCAP     in1 {desc(b)}\nCAP     out {desc(out)}", flush=True)
            for k, v in kw.items():
                print(f"CAP     {k} = {_kw(v)}"[:700], flush=True)
        return out

    ttnn.experimental.minimal_matmul, ttnn.linear = mm, lin
    try:
        generator, model_args, kv_caches, page_table = build_single_device_model(device, batch_size=B, seq_len=512)
        input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, B, 512)
        generator.prefill_forward_text(
            input_ids, page_table=page_table, kv_cache=kv_caches, prompt_lens=prompt_lens, enable_trace=False,
            return_hidden_states=True, warmup_prefill=False,
        )  # fmt: skip
        knobs = sorted(k for k in os.environ if k.startswith(("QWEN_", "TT_PREFILL")))
        print("CAP env " + " ".join(f"{k}={os.environ[k]}" for k in knobs), flush=True)
    finally:
        ttnn.experimental.minimal_matmul, ttnn.linear = orig_mm, orig_lin
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
