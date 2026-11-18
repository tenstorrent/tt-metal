# Device-profile one 4K prefill of a 1-layer TP32 Llama-3.3-70B on the BH galaxy (no warmup sweep, so the profiler buffer
# holds it): prefill once to compile, flush the device profiler, prefill again (the measured one), flush again.
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

SEQ = int(os.environ.get("PREFILL_SEQ", "4096"))


@pytest.mark.parametrize(
    "device_params",
    [
        {
            TRACE_MODEL_KEY_PARAM: "llama3.3-70b-galaxy",
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_prefill_layer_prof(mesh_device, device_params):
    model_args, model, page_table, kv = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=131072,
        num_layers=1,
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 2048},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    model_args.tokenizer = model_args.create_tokenizer()
    gen = Generator(model, model_args, mesh_device, tokenizer=model_args.tokenizer)
    greedy = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    tokens = torch.randint(1000, 100000, (1, SEQ))
    pt = page_table[:1]

    def run():
        out = gen.prefill_forward_text(
            tokens,
            page_table=pt,
            kv_cache=kv,
            prompt_lens=[SEQ],
            enable_trace=False,
            tt_out_logits_all_users=None,
            sampling_params=greedy,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    run()  # compile
    ttnn.ReadDeviceProfiler(mesh_device)  # drain compile-run records
    import time

    t0 = time.perf_counter()
    run()  # measured
    logger.info(
        f"PREFILL_LAYER_PROF seq={SEQ} host wall {1e3 * (time.perf_counter() - t0):.1f} ms for 1 layer (+embed/head)"
    )
    ttnn.ReadDeviceProfiler(mesh_device)
