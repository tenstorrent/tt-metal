# Probe: what keeps the Blackhole prefetcher global CB alive after switching back to prefill?
# prefill -> decode -> prefill(2K). A 2K prefill needs static CBs up to ~1.09 MB, so it only runs if the
# decode global CB was really freed. Retries after gc and after clearing the program cache.
import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM


def _global_cbs():
    return [o for o in gc.get_objects() if type(o).__name__ == "GlobalCircularBuffer"]


def _describe_referrers(obj):
    out = []
    for r in gc.get_referrers(obj):
        if isinstance(r, dict):
            owners = [type(o).__name__ for o in gc.get_referrers(r) if getattr(o, "__dict__", None) is r]
            keys = [k for k, v in r.items() if v is obj]
            out.append(f"dict{keys} of {owners}")
        elif isinstance(r, list) and r is not _global_cbs:
            out.append(f"list(len={len(r)})")
        else:
            out.append(type(r).__name__)
    return out


def _l1_state(mesh_device, label):
    try:
        mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
        logger.info(
            f"CBPROBE L1 [{label}] allocated/bank={mv.total_bytes_allocated_per_bank} "
            f"largest_free/bank={mv.largest_contiguous_bytes_free_per_bank} total/bank={mv.total_bytes_per_bank}"
        )
        blocks = []
        for b in mv.block_table:
            row = dict(b)
            if str(row.get("allocated", "")).lower() in ("yes", "true", "1"):
                blocks.append(row)
        logger.info(f"CBPROBE L1 [{label}] allocated blocks: {blocks[-12:]}")
    except Exception as e:
        logger.info(f"CBPROBE L1 [{label}] memory view failed: {e}")


def _try_prefill(gen, x, pt1, kv, n, greedy, tok, label):
    try:
        t = gen.prefill_forward_text(
            x, page_table=pt1, kv_cache=kv, prompt_lens=[n], enable_trace=False, tt_out_logits_all_users=None,
            sampling_params=greedy,
        )
        first = int(torch.as_tensor(t[0] if isinstance(t, (tuple, list)) else t).flatten()[0])
        logger.info(f"CBPROBE {label}: prefill {n} OK token {tok.decode([first])!r}")
        return True
    except Exception as e:
        logger.info(f"CBPROBE {label}: prefill {n} FAILED {str(e)[:260]}")
        return False


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
def test_cb_free_probe(mesh_device, device_params):
    model_args, model, page_table, kv = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=131072,
        num_layers=80,
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 2048},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    model_args.tokenizer = model_args.create_tokenizer()
    tok = model_args.tokenizer
    gen = Generator(model, model_args, mesh_device, tokenizer=tok)
    greedy = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    pt1 = page_table[:1]

    ids = list(tok.encode("The capital of France is", bos=True, eos=False))
    x_short = torch.tensor([ids], dtype=torch.long)
    filler = list(tok.encode(" The history of Rome spans many centuries.", bos=False, eos=False))
    long_ids = (ids[:1] + filler * 400)[:1990] + ids[1:]
    x_long = torch.tensor([long_ids], dtype=torch.long)

    _try_prefill(gen, x_long, pt1, kv, len(long_ids), greedy, tok, "before any decode")
    logger.info(f"CBPROBE live GlobalCircularBuffer objects after first prefill: {len(_global_cbs())}")

    for cycle in range(2):
        model.switch_mode("decode")
        out_tok = torch.full((32, 1), ids[-1], dtype=torch.long)
        pos = torch.nn.functional.pad(torch.tensor([len(ids)]), (0, 31), value=-1)
        for it in range(2):
            gen.decode_forward(
                out_tok, pos, enable_trace=False, page_table=page_table, kv_cache=kv, read_from_device=True,
                async_read=False, sampling_params=greedy, tt_out_logits_saved=None, is_cur_pos_sharded=False,
                is_page_table_sharded=False, prompt_tokens=x_short, output_tokens=torch.tensor([[ids[-1]]]),
                reload_inputs=True, reload_page_table=False, reload_sampling_params=True,
                reset_sampling_state=(it == 0),
            )
            pos = pos + torch.tensor([1] + [0] * 31)
        _l1_state(mesh_device, f"cycle {cycle} decode mode")
        model.switch_mode("prefill")
        _l1_state(mesh_device, f"cycle {cycle} after switch to prefill")
        if not _try_prefill(gen, x_long, pt1, kv, len(long_ids), greedy, tok, f"cycle {cycle} after switch"):
            break
    else:
        return
    for name in ("clear_program_cache", "disable_and_clear_program_cache"):
        fn = getattr(mesh_device, name, None)
        if fn is not None:
            fn()
            logger.info(f"CBPROBE called mesh_device.{name}()")
            break
    if hasattr(mesh_device, "enable_program_cache"):
        mesh_device.enable_program_cache()
    gc.collect()
    _l1_state(mesh_device, "after clearing program cache")
    _try_prefill(gen, x_long, pt1, kv, len(long_ids), greedy, tok, "after clearing program cache")
