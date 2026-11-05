# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of control flow without importing or opening TT devices."""

import ast
import io
import json
import math
from contextlib import redirect_stdout
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).parents[1]


def load_class(filename, name, namespace):
    tree = ast.parse((ROOT / "tt" / filename).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def main():
    cls = load_class(
        "generator.py",
        "Gemma4Generator",
        {
            "Generator": object,
            "torch": torch,
            "json": json,
            "math": math,
            "Integral": Integral,
            "Real": Real,
            "SamplingParams": SimpleNamespace,
            "format_sampling_params": lambda params, _: params,
        },
    )
    gen = cls.__new__(cls)
    gen.kv_cache, gen.page_table = object(), object()
    gen.prepare_decode = Mock()
    sentinel = object()
    gen._replay = Mock(return_value=sentinel)
    # Non-tensor sentinels fail as_tensor if stale inputs are read at all.
    output = gen.decode_forward(
        object(),
        object(),
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        reload_inputs=False,
        read_from_device=False,
    )
    assert output is sentinel
    gen._replay.assert_called_once_with(host_sampling=False, read_from_device=False)
    # The stock runner supplies NumPy scalar lengths, not always Python ints.
    hidden = object()
    gen.prefill = SimpleNamespace(stage=Mock(), hidden=Mock(return_value=hidden))
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        chunks = list(
            gen._prefill_hidden_chunks(torch.arange(129), slot=np.int64(0), start=np.int64(0), end=np.int64(129))
        )
    assert chunks == [(0, 129, hidden)]
    assert gen.prefill.stage.call_args.kwargs == dict(slot=0, start=0, size=160, valid=129)
    event = json.loads(stdout.getvalue())
    assert event["event"] == "generator_prefill_chunk"
    assert event["valid"] == 129 and event["physical_size"] == 160
    assert all(type(event[key]) is int for key in ("slot", "logical_start", "start", "valid", "physical_size"))
    json.dumps(event)
    gen.batch = 1
    gen.sampler = SimpleNamespace(reset_params=Mock(), seeds_tt_tensor=object())
    gen._copy = Mock()
    gen.set_sampling(seed=[2**31 + 7])
    assert gen.seed_values.dtype == torch.uint32 and int(gen.seed_values[0]) == 2**31 + 7
    assert gen._copy.call_args.args[0].dtype == torch.uint32
    adapter_cls = load_class("generator_vllm.py", "Gemma4ForCausalLM", {"torch": torch, "json": json, "math": math})
    # Pinned worker adds eight sliding windows and one output block per row.
    # Compare physical capacity against five real sliding groups, and against
    # the prior selected-policy allocator budget rather than only the formula.
    pool_checks = []
    expected_blocks = {1: 8449, 8: 10248, 16: 18448, 32: 36896}
    with redirect_stdout(io.StringIO()):
        for batch in (1, 8, 16, 32):
            budget = adapter_cls.get_max_tokens_all_users(max_model_len=262144, max_num_seqs=batch)
            blocks = math.ceil((budget + batch * (32 + 8 * 1024)) / 32)
            assert blocks == expected_blocks[batch]
            working_set_blocks = batch * (math.ceil(32768 / 64) + 5 * math.ceil(1024 / 32))
            assert blocks > working_set_blocks + 5 * math.ceil(4096 / 32) + batch + 1
            assert blocks > math.ceil(262144 / 64) + 5 * math.ceil((1024 + 4096) / 32) + batch + 1
            cache_bytes = blocks * 10 * 2 * 32 * 1088
            prior_weights_rope_state = 6419826688
            measured_prepared_other = 136823296
            reserve = 1024**3
            free_bytes = 34044513280 - cache_bytes - prior_weights_rope_state - measured_prepared_other
            assert free_bytes > reserve
            if batch >= 16:
                # The reserve buys cached-history capacity beyond the live
                # KV bound; it does not establish a hit-rate guarantee.
                assert blocks - working_set_blocks >= batch * math.ceil(24576 / 64)
            pool_checks.append(
                dict(
                    batch=batch,
                    scheduler_blocks=blocks,
                    cache_bytes_per_device=cache_bytes,
                    projected_free_bytes_per_device=free_bytes,
                )
            )
    block_ids = torch.tensor([[7, 2, 11, 0]], dtype=torch.int32)
    assert adapter_cls._physical_pages(block_ids, 2, 129).tolist() == [[14, 15, 4, 5, 22]]
    assert adapter_cls._physical_pages(block_ids, 1, 97).tolist() == [[7, 2, 11, 0]]
    try:
        adapter_cls._physical_pages(torch.tensor([[3, 4, 5]]), 2, 65)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonzero scheduler suffix must not be silently truncated")
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.batch = 8
    adapter.sampler_seeds = [9] * 8
    adapter.generator = SimpleNamespace(set_sampling=Mock())
    params = SimpleNamespace(temperature=[1.0], top_k=[32], top_p=[0.95], seed=[42])
    adapter._sampling(params)
    kw = adapter.generator.set_sampling.call_args.kwargs
    assert kw["temperature"] == [1.0] + [0.0] * 7
    assert kw["top_k"] == [32] + [1] * 7
    assert kw["seed"][0] == 42
    seeds = adapter.sampler_seeds[:]
    adapter._sampling(params, reset_seed=False)
    assert adapter.sampler_seeds == seeds
    assert adapter.generator.set_sampling.call_args.kwargs["reset_seed"] is False
    numpy_params = SimpleNamespace(
        temperature=np.array([1.0, 0.8, 0.0]),
        top_k=np.array([32, 16, 1], dtype=np.int64),
        top_p=np.array([0.95, 0.9, 1.0]),
        seed=np.array([42, 7, 2**31 - 10], dtype=np.int64),
    )
    adapter._sampling(numpy_params, positions=np.array([129, 130, 131], dtype=np.int64))
    numpy_kw = adapter.generator.set_sampling.call_args.kwargs
    assert numpy_kw["temperature"] == [1.0, 0.8, 0.0] + [0.0] * 5
    assert numpy_kw["top_k"] == [32, 16, 1] + [1] * 5
    assert numpy_kw["top_p"] == [0.95, 0.9, 1.0] + [1.0] * 5
    assert numpy_kw["seed"][:3] == [171, 137, 2**31 + 121]
    assert len(numpy_kw["seed"]) == 8 and all(type(seed) is int for seed in numpy_kw["seed"])
    numpy_seeds = adapter.sampler_seeds[:]
    adapter._sampling(numpy_params, reset_seed=False, positions=np.array([4096, 8192, 12288], dtype=np.int64))
    assert adapter.sampler_seeds == numpy_seeds
    assert adapter.generator.set_sampling.call_args.kwargs["reset_seed"] is False
    adapter.sampler_seeds = [2**31 - 1, 2**32 - 1] + [0] * 6
    adapter._advance_sampler()
    assert adapter.sampler_seeds[:2] == [2**31, 0]
    boundary_params = SimpleNamespace(temperature=[0.7], top_k=[32], top_p=[0.9], seed=[2**31 - 130])
    adapter._sampling(boundary_params, positions=[129])
    assert adapter.sampler_seeds[0] == 2**31 - 1
    adapter._advance_sampler()
    assert adapter.sampler_seeds[0] == 2**31
    adapter._sampling(boundary_params, positions=[130])
    assert adapter.sampler_seeds[0] == 2**31
    external_params = SimpleNamespace(temperature=[0.7, 0.7], top_k=[32, 32], top_p=[0.9, 0.9], seed=[-2, 2**40])
    adapter._sampling(external_params, positions=[3, 129])
    assert adapter.sampler_seeds[:2] == [2**31 + 1, 129]
    adapter.kv_cache = object()
    adapter.generator = SimpleNamespace(page_table=object(), decode_forward=Mock(return_value=torch.tensor([[17.0]])))
    adapter._pages = Mock()
    adapter._log = Mock()
    adapter.batch = 1
    adapter.decode_forward(
        torch.tensor([[2]]),
        torch.tensor([129]),
        page_table=sentinel,
        kv_cache=adapter.kv_cache,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    adapter._pages.assert_called_once_with(sentinel, None, adapter.kv_cache)
    adapter._pages.reset_mock()
    adapter.decode_forward(
        object(),
        object(),
        page_table=object(),
        kv_cache=adapter.kv_cache,
        reload_inputs=False,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=False,
    )
    adapter._pages.assert_not_called()
    adapter.sampler_seeds = [2**31 + 7]
    adapter.generator._copy = Mock()
    adapter.generator.sampler = SimpleNamespace(seeds_tt_tensor=object())
    adapter.decode_forward(
        object(),
        object(),
        page_table=object(),
        kv_cache=adapter.kv_cache,
        reload_inputs=False,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=False,
        slot_remap=[0],
    )
    uploaded = adapter.generator._copy.call_args.args[0]
    assert uploaded.dtype == torch.uint32 and int(uploaded[0]) == 2**31 + 7
    adapter.model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))
    valid = torch.tensor([[0], [15]], dtype=torch.int64)
    assert adapter._validate_token_ids(valid) is valid
    assert adapter.device_token_rows_checked == 2
    for invalid in (-1, 16):
        try:
            adapter._validate_token_ids(torch.tensor([[invalid]], dtype=torch.int64))
        except ValueError as error:
            assert "outside vocabulary" in str(error)
        else:
            raise AssertionError("Out-of-vocabulary sample was accepted")
    assert adapter.device_token_rows_checked == 2
    print(
        json.dumps(
            {
                "scope": "host-only source control flow; no TTNN import or device use",
                "stale_inputs_not_read": True,
                "partial_batch_parameter_padding": True,
                "parameter_reload_preserves_rng": True,
                "uint32_seed_rollover_and_remap": True,
                "scheduler_block_expansion_32_64": True,
                "full_reload_includes_pages": True,
                "steady_decode_skips_pages": True,
                "numpy_prefill_lengths_json_serializable": True,
                "numpy_sampling_arrays_and_position_seed_offsets": True,
                "normalized_pool_capacity_bounds": pool_checks,
                "sampled_token_range_audit_rejects_invalid_ids": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
