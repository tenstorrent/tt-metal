# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks: run this file directly to avoid device pytest fixtures."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tt" / "precision.py"
SPEC = importlib.util.spec_from_file_location("qwen_precision_cpu", SOURCE)
precision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precision)


class PrecisionPolicyTest(unittest.TestCase):
    def test_static_weight_delta_baseline_and_edge_experts(self):
        policy = precision.inherited_policy()
        self.assertEqual(precision.static_weight_delta(policy), 0)
        policy["layer_exceptions"] = {
            str(index): {"weight_groups": {"expert_" + name: "bfloat8_b" for name in ("gate", "up", "down")}}
            for index in (0, 47)
        }
        snapshot = copy.deepcopy(policy)
        # EP4 leaves 128 matrices/role/rank, each 2560 by 640, with 32-square tiles.
        expected = 2 * 128 * 3 * (2560 // 32) * (640 // 32) * (1088 - 576)
        self.assertEqual(precision.static_weight_delta(policy), expected)
        self.assertEqual(expected, 629145600)
        self.assertEqual(policy, snapshot)
        policy["layer_exceptions"] = {}
        policy["weight_groups"].update({"expert_" + name: "bfloat8_b" for name in ("gate", "up", "down")})
        self.assertEqual(precision.static_weight_delta(policy), 15099494400)

    def test_static_weight_delta_projection_shapes_and_bank_padding(self):
        # Derive tiled storage from the actual fixed projection geometries.
        def banked(k, widths):
            columns = sum((n + 31) // 32 for n in widths)
            return (k // 32) * ((columns + 7) // 8) * 8

        counts = {
            "shared": 48 * banked(2560, [160, 160]),
            "shared_down": 48 * ((160 // 32) * (2560 // 32)),
            "qsa_qkv": 12 * banked(2560, [3072, 256, 256]),
            "qsa_out": 12 * banked(1536, [2560]),
            "gr": 48 * 2 * banked(10240, [1]),
            "gdn_zab": 36 * banked(2560, [1536 + 12 + 12]),
            "lm_head": banked(2560, [62080]),
        }
        tiles = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}
        for group, count in counts.items():
            with self.subTest(group=group):
                policy = precision.inherited_policy()
                before = policy["weight_groups"][group]
                policy["weight_groups"][group] = "bfloat4_b"
                self.assertEqual(precision.static_weight_delta(policy), count * (576 - tiles[before]))
        self.assertEqual(counts["lm_head"], 155520)

    def test_explicit_policy_and_layer_resolution_do_not_mutate_inputs(self):
        source = precision.inherited_policy()
        source["layer_exceptions"] = {
            "0": {"weight_groups": {"expert_gate": "bfloat8_b", "expert_up": "bfloat8_b", "expert_down": "bfloat8_b"}},
            "47": {"compute_fidelities": {"experts": "HiFi2"}},
        }
        snapshot = copy.deepcopy(source)
        loaded = precision.load_policy("unused", source)
        first = precision.layer_policy(loaded, 0)
        last = precision.layer_policy(loaded, 47)
        middle = precision.layer_policy(loaded, 24)
        self.assertEqual(first["weight_groups"]["expert_down"], "bfloat8_b")
        self.assertEqual(last["compute_fidelities"]["experts"], "HiFi2")
        self.assertEqual(middle["weight_groups"]["expert_down"], "bfloat4_b")
        first["fixed_numerics"]["router_weights"] = "changed"
        first["layer_exceptions"]["47"]["compute_fidelities"]["experts"] = "LoFi"
        self.assertEqual(source, snapshot)
        self.assertEqual(loaded, snapshot)
        self.assertEqual(last["compute_fidelities"]["experts"], "HiFi2")

    def test_selected_default_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as root:
            selected = Path(root) / "doc/datatype_sweep/selected_precision_config.json"
            selected.parent.mkdir(parents=True)
            policy = precision.inherited_policy()
            policy["config_id"] = "selected_test"
            policy["compute_fidelities"]["shared"] = "LoFi"
            selected.write_text(json.dumps(policy))
            self.assertEqual(precision.load_policy(root), policy)
            self.assertEqual(precision.load_policy(root, "doc/datatype_sweep/selected_precision_config.json"), policy)
            explicit = precision.inherited_policy()
            self.assertEqual(precision.load_policy(root, explicit), explicit)

    def test_missing_and_unknown_fields_are_rejected(self):
        for section in (None, "weight_groups", "compute_fidelities"):
            for mutation in ("remove", "unknown"):
                with self.subTest(section=section, mutation=mutation):
                    policy = precision.inherited_policy()
                    target = policy if section is None else policy[section]
                    if mutation == "remove":
                        del target[next(iter(target))]
                    else:
                        target["typo"] = "bfloat16"
                    with self.assertRaises(ValueError):
                        precision.load_policy("unused", policy)

    def test_unsupported_dtype_fidelity_and_fixed_contract_are_rejected(self):
        cases = [
            ("weight_groups", "shared", "float32"),
            ("compute_fidelities", "shared", "HiFi3"),
            (None, "activation_dtype", "bfloat4_b"),
            (None, "ccl_dtype", "float32"),
            (None, "kv_cache_dtype", "bfloat4_b"),
            (None, "residual_dtype", "bfloat8_b"),
            (None, "logits_dtype", "float32"),
            (None, "sampling_dtype", "float32"),
            (None, "max_context", 32768),
            (None, "kv_layout", "page64"),
            ("fixed_numerics", "router_logits", "bfloat16"),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                policy = precision.inherited_policy()
                (policy if section is None else policy[section])[key] = value
                with self.assertRaises(ValueError):
                    precision.load_policy("unused", policy)

    def test_invalid_layer_keys_and_overrides_are_rejected(self):
        overrides = [
            {"-1": {}},
            {"48": {}},
            {"01": {}},
            {"1.0": {}},
            {"0": {"activation_dtype": "bfloat8_b"}},
            {"0": {"weight_groups": {"shared_typo": "bfloat4_b"}}},
            {"0": {"weight_groups": {"shared": "float32"}}},
            {"0": {"compute_fidelities": {"shared": "HiFi3"}}},
        ]
        for override in overrides:
            with self.subTest(override=override):
                policy = precision.inherited_policy()
                policy["layer_exceptions"] = override
                with self.assertRaises(ValueError):
                    precision.load_policy("unused", policy)

    def test_global_lm_head_cannot_be_silently_overridden_per_decoder_layer(self):
        # QwenModel constructs the LM head once from the top-level policy.
        for group, value in (("weight_groups", "bfloat8_b"), ("compute_fidelities", "LoFi")):
            with self.subTest(group=group):
                policy = precision.inherited_policy()
                policy["layer_exceptions"] = {"0": {group: {"lm_head": value}}}
                with self.assertRaises(ValueError):
                    precision.load_policy("unused", policy)

    def test_incompatible_fused_expert_dtypes_rejected_before_model_load(self):
        # The fused kernel has one weight format across gate/up/down.
        for per_layer in (False, True):
            with self.subTest(per_layer=per_layer):
                policy = precision.inherited_policy()
                if per_layer:
                    policy["layer_exceptions"] = {"0": {"weight_groups": {"expert_down": "bfloat8_b"}}}
                else:
                    policy["weight_groups"]["expert_down"] = "bfloat8_b"
                with self.assertRaises(ValueError):
                    precision.load_policy("unused", policy)


if __name__ == "__main__":
    unittest.main()
