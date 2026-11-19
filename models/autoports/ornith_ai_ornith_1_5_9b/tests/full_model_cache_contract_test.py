# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only cache/budget/lifecycle protocol tests; no device fixtures/imports."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import (
    FULL_LAYERS,
    PERSISTENT_CATEGORIES,
    CacheOwnershipRegistry,
    DramBudget,
    get_kv_cache_spec,
    get_max_tokens_all_users,
    plan_cache_allocation,
    plan_cache_specs,
    storage_bytes,
)


def model_and_cache(*, slots=3, context=256, blocks=29, global_pages=True):
    model = NS(
        device=NS(shape=(1, 4)),
        layer_indices=list(range(32)),
        config=NS(max_position_embeddings=262144),
        layers=[
            NS(kind="full_attention" if i in FULL_LAYERS else "linear_attention", kv_dtype="DataType.BFLOAT8_B")
            for i in range(32)
        ],
    )

    def tensor(shape, dtype):
        return NS(shape=shape, dtype=dtype, owner=model.device)

    cache = []
    for layer in model.layers:
        conv = recurrent = key = value = None
        if layer.kind == "full_attention":
            key, value = [tensor((blocks, 1, 32, 256), "bfloat8_b") for _ in range(2)]
        else:
            conv = tensor((3, 1, slots, 2048), "bfloat16")
            recurrent = tensor((slots, 8, 128, 128), "float32")
        cache.append(
            [
                NS(
                    key=key,
                    value=value,
                    conv=conv,
                    recurrent=recurrent,
                    state_row=slot if conv is not None else None,
                    page_offset=slot * ((context + 31) // 32) if key is not None else 0,
                    page_count=(context + 31) // 32 if key is not None else None,
                    global_pages=global_pages if key is not None else False,
                )
                for slot in range(slots)
            ]
        )
    return model, cache


class CacheContractTests(unittest.TestCase):
    def budget(self, **changes):
        args = {
            "measured_free_bytes_per_device": (20_000_000, 21_000_000, 19_000_000, 20_000_000),
            "measurement_artifact": "measured_four_devices.json",
            "included_categories": tuple(PERSISTENT_CATEGORIES),
            "pending_bytes_per_device": {"warmup_peak": 1_000_000},
            "reserve_bytes_per_device": 2_000_000,
        }
        args.update(changes)
        return DramBudget(**args)

    def test_exact_tiled_precision_and_spec_split(self):
        specs = get_kv_cache_spec()
        self.assertEqual(tuple(specs), FULL_LAYERS)
        self.assertEqual({x.group_index for x in specs.values()}, {0})
        first = specs[3]
        self.assertEqual(first.key_bytes_per_block, 8 * 1088)
        self.assertEqual(first.value_bytes_per_block, 8 * 1088)
        self.assertEqual(first.prefill_input_dtype, "bfloat16")
        self.assertEqual(storage_bytes((1, 1, 32, 256), "bfloat16"), 8 * 2048)
        self.assertEqual(storage_bytes((3, 1, 3, 2048), "bfloat16"), 3 * 64 * 2048)
        with self.assertRaises(ValueError):
            storage_bytes((1, 32), "bfloat8_b", tiled=False)

    def test_actual_k_and_v_checked_independently(self):
        model, cache = model_and_cache()
        self.assertEqual(get_kv_cache_spec(model, cache=cache)[3].bytes_per_block, 17408)
        cache[3][0].value.dtype = "bfloat16"
        with self.assertRaises(ValueError):
            get_kv_cache_spec(model, cache=cache)
        model.layer_indices = [0, 3]
        with self.assertRaises(ValueError):
            get_kv_cache_spec(model)

    def test_bf16_policy_counts_both_cache_tensors_at_actual_precision(self):
        model, cache = model_and_cache()
        for index in FULL_LAYERS:
            model.layers[index].kv_dtype = "bfloat16"
            cache[index][0].key.dtype = "bfloat16"
            cache[index][0].value.dtype = "bfloat16"
        specs = get_kv_cache_spec(model, cache=cache)
        self.assertEqual(specs[3].bytes_per_block, 32768)
        self.assertEqual(sum(s.bytes_per_block for s in specs.values()), 262144)
        cache[3][0].value.dtype = "bfloat8_b"
        with self.assertRaises(ValueError):
            get_kv_cache_spec(model, cache=cache)

    def test_budget_uses_weakest_device_and_explicit_peak_reserve(self):
        limit = self.budget().pool_limit()
        self.assertEqual(limit.bytes_per_block_per_device, 8 * 2 * 8 * 1088)
        self.assertEqual(limit.usable_bytes_per_device, (17_000_000, 18_000_000, 16_000_000, 17_000_000))
        self.assertEqual(limit.num_blocks_upper_bound, 16_000_000 // 139264)
        self.assertFalse(limit.allocation_verified)
        for changes in (
            {"included_categories": ("weights",)},
            {"reserve_bytes_per_device": 0},
            {"measurement_artifact": ""},
            {"pending_bytes_per_device": {"weights": 3}},
            {"measured_free_bytes_per_device": (20_000_000,)},
        ):
            with self.assertRaises(ValueError):
                self.budget(**changes)

    def test_no_unmeasured_capacity_or_silent_context_reduction(self):
        args = {
            "budget": self.budget(),
            "validated_num_blocks": 80,
            "validation_artifact": "allocation_and_peak.json",
            "max_model_len": 256,
            "max_num_seqs": 32,
        }
        self.assertEqual(get_max_tokens_all_users(**args), 80 * 32)
        for changes in (
            {"validation_artifact": ""},
            {"validated_num_blocks": 1000},
            {"validated_num_blocks": 7},
            {"num_devices": 1},
            {"max_num_seqs": 33},
        ):
            with self.assertRaises(ValueError):
                get_max_tokens_all_users(**(args | changes))
        self.assertEqual(get_max_tokens_all_users(**(args | {"max_num_seqs": 2})), 2 * 256)

    def test_arbitrary_global_pool_registers_exact_backing_and_counts(self):
        model, cache = model_and_cache(blocks=29)
        registry = CacheOwnershipRegistry(model, tensor_owner=lambda tensor: tensor.owner)
        binding = registry.register(cache, context=256, global_pages=True, num_blocks=29)
        self.assertEqual(binding.num_blocks, 29)
        self.assertEqual(binding.kv_bytes_per_device, 29 * 139264)
        self.assertEqual(binding.recurrent_bytes_per_device, 24 * (3 * 64 * 2048 + 3 * 8 * 128 * 128 * 4))
        self.assertIs(binding, registry.register(cache, context=256, global_pages=True, num_blocks=29))
        self.assertIs(binding.cache, cache)

    def test_private_geometry_alias_owner_and_policy_rejections(self):
        for mutate in (
            lambda m, c: setattr(c[3][0].value, "dtype", "bfloat16"),
            lambda m, c: setattr(c[7][0], "key", c[3][0].key),
            lambda m, c: setattr(c[0][1], "state_row", 0),
            lambda m, c: setattr(c[3][0].key, "shape", (28, 1, 32, 256)),
            lambda m, c: setattr(c[3][0].key, "owner", object()),
            lambda m, c: c[1].pop(),
        ):
            model, cache = model_and_cache()
            mutate(model, cache)
            with self.assertRaises(ValueError):
                CacheOwnershipRegistry(model, tensor_owner=lambda tensor: tensor.owner).register(
                    cache, context=256, global_pages=True, num_blocks=29
                )
        model, cache = model_and_cache(blocks=24, global_pages=False)
        registry = CacheOwnershipRegistry(model)
        registry.register(cache, context=256, global_pages=False)
        cache[3][1].page_offset = 0
        with self.assertRaises(ValueError):
            registry.register(cache, context=256, global_pages=False)

    def test_release_routes_external_serving_cache_and_seal_forbids_rebinding(self):
        model, own = model_and_cache(blocks=24, global_pages=False)
        _, external = model_and_cache(blocks=29)
        registry = CacheOwnershipRegistry(model)
        registry.register(own, context=256, global_pages=False)
        registry.register(external, context=256, global_pages=True, num_blocks=29)
        registry.bind_serving(external)
        registry.seal()
        calls = []
        registry.release_request(2, reset_slots=lambda slots, **kwargs: calls.append((slots, kwargs)))
        self.assertEqual(calls[0][0], [2])
        self.assertIs(calls[0][1]["kv_cache"], external)
        self.assertIs(calls[0][1]["clear_kv"], False)
        with self.assertRaises(RuntimeError):
            registry.bind_serving(own)
        with self.assertRaises(ValueError):
            registry.release_request(3, reset_slots=lambda *a, **kw: self.fail("must not mutate"))
        _, third = model_and_cache(blocks=30)
        with self.assertRaises(RuntimeError):
            registry.register(third, context=256, global_pages=True, num_blocks=30)

    def test_allocation_plan_honors_passed_physical_blocks(self):
        plan = plan_cache_allocation(batch=3, context=255, num_blocks=29, budget=self.budget())
        self.assertEqual(plan.model_kwargs(), {"batch": 3, "context": 255, "global_pages": True, "num_blocks": 29})
        self.assertEqual(plan.estimated_kv_bytes_per_device, 29 * 139264)
        for blocks in (0, 7, 9999, 8.5, True):
            with self.assertRaises(ValueError):
                plan_cache_allocation(batch=3, context=256, num_blocks=blocks, budget=self.budget())

    def test_pinned_plugin_triples_full_context_and_arbitrary_pool(self):
        # MODEL_SETUP.md / model_runner._build_per_layer_specs single-group
        # output: one triple per ATTENTION layer (not one per K/V tensor).
        triples = [((8192, 1, 32, 256), "DataType.BFLOAT8_B", index) for index in range(8)]
        plan = plan_cache_specs(triples, batch=1, context=262144)
        self.assertEqual(plan.model_kwargs(), {"batch": 1, "context": 262144, "global_pages": True, "num_blocks": 8192})
        self.assertEqual(plan.attention_tensor_indices, tuple(range(8)))
        self.assertEqual(plan.estimated_kv_bytes_per_device, 8192 * 8 * 2 * 8 * 1088)
        # Scheduler identities need not be consecutive or in numeric order.
        identities = (17, 6, 5, 3, 2, 11, 19, 0)
        triples = [((29, 1, 32, 256), NS(name="BFLOAT8_B"), index) for index in identities]
        plan = plan_cache_specs(triples, batch=3, context=255, budget=self.budget())
        self.assertEqual(plan.model_kwargs(), {"batch": 3, "context": 255, "global_pages": True, "num_blocks": 29})
        self.assertEqual(plan.attention_tensor_indices, identities)
        model, cache = model_and_cache(context=255, blocks=plan.num_blocks)
        binding = CacheOwnershipRegistry(model).register_inferred(cache, context=255)
        self.assertEqual(binding.kv_bytes_per_device, plan.estimated_kv_bytes_per_device)
        self.assertEqual(
            len({id(t) for layer in FULL_LAYERS for t in (cache[layer][0].key, cache[layer][0].value)}), 16
        )

    def test_plugin_triples_reject_aliases_precision_geometry_and_missing_layers(self):
        triples = [((29, 1, 32, 256), "bfloat8_b", index) for index in range(8)]
        malformed = [triples[:7], triples + [triples[0]], {index: value for index, value in enumerate(triples)}]
        for replacement in (
            ((29, 1, 32, 256), "bfloat8_b", 0),  # aliases layer0
            ((29, 1, 32, 256), "bfloat16", 1),
            ((29, 4, 32, 256), "bfloat8_b", 1),
            ((29, 1, 16, 256), "bfloat8_b", 1),
            ((29, 1, 32, 128), "bfloat8_b", 1),
            ((30, 1, 32, 256), "bfloat8_b", 1),
            ((29, 1, 32, 256), "bfloat8_b", -1),
            ((29, 1, 32, 256), "bfloat8_b", True),
            ((29, 1, 32, 256), "bfloat8_b", 1.0),
            ((29, True, 32, 256), "bfloat8_b", 1),
            ((29, 1, 32, 256), "bfloat8_b"),
            None,
        ):
            changed = list(triples)
            changed[1] = replacement
            malformed.append(changed)
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_cache_specs(value, batch=3, context=255)
        for changes in ({"batch": 33}, {"context": 262145}, {"context": 1024}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_cache_specs(triples, **({"batch": 3, "context": 255} | changes))
        with self.assertRaises(ValueError):
            plan_cache_specs(
                [((9999, 1, 32, 256), "bfloat8_b", i) for i in range(8)], batch=3, context=255, budget=self.budget()
            )

    def test_no_device_packages_imported(self):
        # Repository pytest fixtures may already import device packages. Isolate
        # the production module import, including its transitive dependencies.
        # -I/-S exclude inherited Python paths and site initialization; only the
        # explicit checkout root is added, so this check also works outside cwd.
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "__import__(sys.argv[2]); "
            "found = sorted(name for name in sys.modules "
            "if name.split('.')[0] in ('ttnn', 'torch', 'vllm')); "
            "assert not found, 'Host-only module imported device packages: ' + repr(found)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                probe,
                str(Path(__file__).resolve().parents[4]),
                "models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
