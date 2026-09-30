# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-model Galaxy trace lifecycle; requires HF_MODEL and real cached weights."""

import os
import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.sampling import SamplingParams
from models.common.utility_functions import comp_pcc
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.demo.text_qwen_demo import create_tt_qwen_model
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import IS_BLACKHOLE, PREFILL_FABRIC_CONFIG
from models.demos.llama3_70b_galaxy.tests.unit_tests.test_prepared_trace_io import _release
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": PREFILL_FABRIC_CONFIG,
            "worker_l1_size": 1345000,
            "l1_small_size": 16384 if IS_BLACKHOLE else 0,
            "trace_region_size": 384000000,
        }
    ],
    indirect=True,
)
def test_full_model_prepared_trace_io(mesh_device, reset_seeds):
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    model_path = os.environ["HF_MODEL"]
    qwen = "Qwen3-32B" in model_path
    assert qwen or "Llama-3.3-70B-Instruct" in model_path
    create = create_tt_qwen_model if qwen else create_tt_model
    layers = 64 if qwen else 80
    args, model, page_table, kv_cache = create(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.accuracy if qwen else LlamaOptimizations.performance,
        max_seq_len=2048,
        num_layers=layers,
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 1024},
        use_paged_kv_cache=True,
    )
    assert len(model.layers) == layers
    expected_prefetcher = not args.is_blackhole or (qwen and os.environ.get("QWEN_BH_PREFETCHER", "0") == "1")
    assert model.use_prefetcher == expected_prefetcher
    tokenizer = args.create_tokenizer()
    generator = Generator(model, args, mesh_device, tokenizer=tokenizer)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    steps = 8
    cases = []
    for lengths in ([97], [777], [97] * 32, [97, 777]):
        prompt = torch.zeros((len(lengths), max(lengths)), dtype=torch.int32)
        for slot, length in enumerate(lengths):
            ids = tokenizer.encode(f"Request {slot}. Explain how computers store and process information. " * 120)
            assert len(ids) >= length
            prompt[slot, :length] = torch.tensor(ids[:length], dtype=torch.int32)
        cases.append((prompt, torch.tensor(lengths, dtype=torch.int32)))

    def prefill(prompt, lengths, trace):
        return generator.prefill_forward_text(
            prompt,
            prompt_lens=lengths,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=trace,
        )

    def start_inputs(logits, lengths):
        tokens = torch.zeros((32, 1), dtype=torch.int32)
        tokens[: len(lengths), 0] = logits[:, 0, :].argmax(-1).to(torch.int32)
        positions = torch.full((32,), -1, dtype=torch.int32)
        positions[: len(lengths)] = lengths
        return tokens, positions

    def ids():
        return (dict(generator.trace_id_prefill), dict(generator.trace_ids_decode), tuple(model.sampling.trace_ids))

    def assert_logits(actual, expected, batch):
        # Original-source Galaxy eager repeats vary in both long and batched
        # prefill and subsequent decode. Compare each user's logits numerically;
        # keep the chosen tokens and complete generated sequences exact.
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        correlations = []
        for slot in range(batch):
            passed, correlation = comp_pcc(expected[slot], actual[slot], 0.99)
            assert passed, f"Logits slot {slot}: {correlation}"
            correlations.append(correlation)
        logger.info(f"TRACE_IO_LOGITS batch={batch} min_pcc={min(correlations)}")
        torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1), rtol=0, atol=0)

    try:
        # Follow the public eager warmup contract before computing independent
        # references. All eager input caches must exist before trace capture too.
        generator.warmup_model_prefill(kv_cache, enable_trace=False, can_sample_on_device=True)
        references = []
        for prompt, lengths in cases:
            logits = prefill(prompt, lengths, False)
            tokens, positions = start_inputs(logits, lengths)
            decode_logits = generator.decode_forward(
                tokens,
                positions,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=False,
            )[0][: len(lengths)].clone()
            sequence = []
            for step in range(steps):
                sampled = (
                    generator.decode_forward(
                        tokens,
                        positions,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        enable_trace=False,
                        sampling_params=params,
                        reset_inputs=True,
                        reset_batch=step == 0,
                        prompt_tokens=prompt,
                        output_tokens=tokens,
                    )[0]
                    .reshape(-1)[: len(lengths)]
                    .clone()
                )
                sequence.append(sampled)
                tokens[: len(lengths), 0] = sampled.to(torch.int32)
                positions[: len(lengths)] += 1
            references.append((logits, decode_logits, torch.stack(sequence)))
        generator.already_warmed_up_prefill = False

        # Public entry points cover both orders across the two full models.
        if qwen:
            generator.warmup_model_prefill(kv_cache, enable_trace=True, can_sample_on_device=True)
            assert not any(generator.trace_ids_decode.values())
        else:
            generator.decode_forward(
                torch.zeros((32, 1), dtype=torch.int32),
                torch.full((32,), -1, dtype=torch.int32),
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=params,
                reset_inputs=True,
                reset_batch=True,
                is_cur_pos_sharded=True,
                is_page_table_sharded=True,
            )
            assert not any(generator.trace_id_prefill.values())
        expected_ids = None
        expected_addresses = None
        expected_cache = None
        expected_layout = None
        for repetition in range(3):
            started = time.perf_counter()
            for (prompt, lengths), (expected_prefill, expected_logits, expected_sequence) in zip(cases, references):
                for sharded in (False, True):
                    logger.info(
                        f"TRACE_IO_REQUEST repetition={repetition} lengths={lengths.tolist()} sharded={sharded}"
                    )
                    actual_prefill = prefill(prompt, lengths, True)
                    assert_logits(actual_prefill, expected_prefill, len(lengths))
                    tokens, positions = start_inputs(actual_prefill, lengths)
                    actual_logits = generator.decode_forward(
                        tokens,
                        positions,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        is_cur_pos_sharded=sharded,
                        is_page_table_sharded=sharded,
                    )[0][: len(lengths)]
                    assert_logits(actual_logits, expected_logits, len(lengths))
                    pending = []
                    sequence = []
                    for step in range(steps):
                        pending.append(
                            generator.decode_forward(
                                tokens,
                                positions,
                                page_table=page_table,
                                kv_cache=kv_cache,
                                sampling_params=params,
                                reset_inputs=step == 0,
                                reset_batch=step == 0,
                                prompt_tokens=prompt,
                                output_tokens=tokens,
                                is_cur_pos_sharded=sharded,
                                is_page_table_sharded=sharded,
                                read_from_device=True,
                                async_read=True,
                            )
                        )
                        if len(pending) > 1:
                            host, events = pending.pop(0)
                            ttnn.event_synchronize(events[0])
                            sequence.append(generator.process_decode_output_host(host)[0].reshape(-1)[: len(lengths)])
                    for host, events in pending:
                        ttnn.event_synchronize(events[0])
                        sequence.append(generator.process_decode_output_host(host)[0].reshape(-1)[: len(lengths)])
                    torch.testing.assert_close(torch.stack(sequence), expected_sequence, rtol=0, atol=0)
            addresses = tuple(t.buffer_address() for *_, t in generator._prepared_trace_io._buffers)
            cache = mesh_device.num_program_cache_entries()
            layout = model.global_cb_trace_state._layout
            if repetition:
                assert ids() == expected_ids
                assert addresses == expected_addresses
                assert cache == expected_cache
                assert layout == expected_layout
            expected_ids, expected_addresses, expected_cache, expected_layout = ids(), addresses, cache, layout
            logger.info(
                f"FULL_MODEL_TRACE_IO layers={layers} repetition={repetition} "
                f"requests=8 decode_steps={8 * steps} active_decode_tokens={2 * 36 * steps} "
                f"seconds={time.perf_counter() - started:.3f} "
                f"programs={cache} buffers={len(addresses)} gcb={layout}"
            )
    finally:
        _release(generator)
