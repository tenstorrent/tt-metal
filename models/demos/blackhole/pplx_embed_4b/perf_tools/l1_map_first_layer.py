# Live L1 buffers after each op of the first decoder layer (one eager prefill at the batch's perf config): lowest L1
# buffer address, total per bank, and the buffer list, next to the device's CB-region start. The room a program's
# static CBs have is [CB region start, lowest live L1 buffer). Usage: l1_map_first_layer.py [batch] [n_ops]
import json
import os
import sys

# the post-operation hooks only fire outside fast runtime mode
os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps({"enable_fast_runtime_mode": False})

from ttnn.decorators import register_post_operation_hook  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.pplx_embed_4b.demo._common import (  # noqa: E402
    apply_workload_env,
    build_single_device_model,
    generate_synthetic_inputs,
)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_OPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def main():
    apply_workload_env(B, 512)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000, num_command_queues=1)
    try:
        generator, model_args, kv_caches, page_table = build_single_device_model(device, batch_size=B, seq_len=512)
        input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, B, 512)
        info = ttnn._ttnn.reports.get_device_info(device)
        print(
            f"L1MAP device: worker_l1_size {info.worker_l1_size} l1_bank_size {info.l1_bank_size} "
            f"first_l1_bank {info.address_at_first_l1_bank} first_cb {info.address_at_first_l1_cb_buffer} "
            f"cb_limit {info.cb_limit} l1_banks {info.l1_num_banks}",
            flush=True,
        )
        n = [0]

        def hook(operation, function_args, function_kwargs, output):
            if n[0] >= N_OPS:
                return None
            n[0] += 1
            ttnn.synchronize_device(device)
            l1 = sorted(
                (b.address, b.max_size_per_bank)
                for b in ttnn._ttnn.reports.get_buffers([device])
                if int(getattr(b.buffer_type, "value", b.buffer_type)) == 1
            )
            low = l1[0][0] if l1 else None
            name = getattr(operation, "python_fully_qualified_name", str(operation))[:48]
            bufs = " ".join(f"{a // 1024}K+{s // 1024}K" for a, s in l1)
            print(f"L1MAP op {n[0]:3d} {name:48s} low {low} total {sum(s for _, s in l1)} bufs {bufs}", flush=True)
            return None

        with register_post_operation_hook(hook):
            generator.prefill_forward_text(
                input_ids, page_table=page_table, kv_cache=kv_caches, prompt_lens=prompt_lens, enable_trace=False,
                return_hidden_states=True, warmup_prefill=False,
            )  # fmt: skip
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
