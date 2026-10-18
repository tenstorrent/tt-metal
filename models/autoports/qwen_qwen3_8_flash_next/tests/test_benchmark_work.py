# SPDX-License-Identifier: Apache-2.0
"""Host-only work-accounting checks; intentionally never imports TTNN."""

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "benchmark_work.py"
SPEC = importlib.util.spec_from_file_location("benchmark_work_host_test", SOURCE)
WORK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORK)


class BenchmarkWorkTests(unittest.TestCase):
    @staticmethod
    def resident_provenance():
        return copy.deepcopy(WORK.RESIDENT_PREFILL_PROVENANCE)

    def v14_ready(self):
        ready = self.v13_ready()
        ready["execution_version"] = WORK.RESIDENT_PREFILL_VERSION
        indices = [i for i in range(48) if i % 4 != 3]
        flags = lambda value: [dict(layer=i, enabled=value) for i in indices]
        ready["gdn_prefill_resident_enabled"] = dict(
            owner=flags(False), prefill=flags(True), decode={str(b): flags(False) for b in (4, 8, 16)}
        )
        ready["prefill_view"]["gdn_prefill_resident"] = [
            dict(
                layer=i,
                enabled=True,
                owner_enabled=False,
                signatures=[
                    dict(
                        sequence=seq,
                        batch=1,
                        workers=96,
                        output_columns=16,
                        local_tile_shape=[32, 32],
                        dst_tiles=8,
                        dst_sync="full",
                        state_tiles=4,
                        state_resident=True,
                        compute_kernel_config=dict(
                            math_fidelity="MathFidelity.HiFi4",
                            fp32_dest_acc_en=True,
                            math_approx_mode=False,
                            dst_full_sync_en=True,
                        ),
                        reader_variant="scalar_cache_narrow_rm",
                        scalar_cache_cb=16,
                        scalar_cache_bytes_per_worker=8192,
                        qk_read_bytes_per_token_per_job=[512, 512],
                        v_read_bytes_per_token_per_job=64,
                        scalar_read_bytes_per_32_tokens_per_job=8192,
                        initial_state_reads_per_job=1,
                        final_state_writes_per_job=1,
                        provenance=copy.deepcopy(self.resident_provenance()),
                    )
                    for seq in (128, 256, 512)
                ],
            )
            for i in indices
        ]
        return ready

    def test_v14_pins_exact_sources_and_consumes_real_signature_function_without_ttnn(self):
        helper = SOURCE.parents[1] / "tt" / "gdn_prefill_resident.py"
        module = ast.parse(helper.read_text())
        literals = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("READER", "COMPUTE", "NATIVE_CONTRACT_SHA256")
        }
        provenance = dict(
            helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            reader_sha256=hashlib.sha256(literals["READER"].encode()).hexdigest(),
            compute_sha256=hashlib.sha256(literals["COMPUTE"].encode()).hexdigest(),
            native_contract_sha256=literals["NATIVE_CONTRACT_SHA256"],
        )
        self.assertEqual(provenance, WORK.RESIDENT_PREFILL_PROVENANCE)
        self.assertEqual(len(provenance["native_contract_sha256"]), 7)
        repo = SOURCE.parents[4]
        for name, digest in provenance["native_contract_sha256"].items():
            self.assertEqual(hashlib.sha256((repo / name).read_bytes()).hexdigest(), digest, name)
        function = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_record_signature"
        )
        namespace = {"_PROVENANCE": provenance}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(helper), "exec"), namespace)
        owner = SimpleNamespace(prefill_resident_signatures={})
        cb = lambda size, indices: SimpleNamespace(
            total_size=size,
            format_descriptors=[SimpleNamespace(buffer_index=index, page_size=4096) for index in indices],
        )
        config = SimpleNamespace(
            math_fidelity="MathFidelity.HiFi4", fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
        )
        for length in (128, 256, 512):
            namespace["_record_signature"](owner, 1, length, 96, config, [cb(8192, [16]), cb(16384, [0, 12])])
        ready = self.v14_ready()
        for row in ready["prefill_view"]["gdn_prefill_resident"]:
            self.assertEqual(row["signatures"], list(owner.prefill_resident_signatures.values()))
        WORK.validate_resident_prefill_ready(ready)
        # These are load/store epochs; each consists of eight1024-byte face transfers.
        self.assertIn("for(uint32_t i=0;i<4;++i) for(uint32_t f=0;f<2;++f)", literals["READER"])
        self.assertIn("if(token+1==length)", literals["READER"])

    def test_v14_preserves_v13_roofline_numerators_and_native_b1(self):
        with patch.object(WORK, "RESIDENT_PREFILL_PROVENANCE", self.resident_provenance()):
            for length in (127, 128, 129, 256, 512, 1024, 4096):
                event = self.v13_prefill([71], [length])
                old = WORK.account_events([self.v13_ready(), event])
                new = WORK.account_events([self.v14_ready(), event])
                with self.subTest(length=length):
                    self.assertEqual(new["prefill"], old["prefill"])
                    self.assertEqual(new["decode"], old["decode"])
                    self.assertEqual(
                        new["details"]["prefill_useful_matrix_flops_by_fidelity"],
                        old["details"]["prefill_useful_matrix_flops_by_fidelity"],
                    )
            for batch in (4, 8, 16, 32):
                event = self.v5_step(batch, batch)
                old = WORK.account_events([self.v13_ready(), event])
                new = WORK.account_events([self.v14_ready(), event])
                self.assertEqual(new["decode"], old["decode"])
                self.assertEqual(
                    new["details"]["decode_dram_bytes_by_category"], old["details"]["decode_dram_bytes_by_category"]
                )
        native = self.v7_ready(batch=1)
        event = dict(phase="decode", physical_batch=1, positions=[4096], suffix_lengths=[1])
        historical = WORK.account_events([native, event])
        flags = [dict(layer=i, enabled=False) for i in range(48) if i % 4 != 3]
        native["gdn_prefill_resident_enabled"] = dict(owner=flags, prefill=None, decode={})
        self.assertEqual(WORK.account_events([native, event]), historical)

    def test_v14_reader_payload_matches_actual_fragments(self):
        with patch.object(WORK, "RESIDENT_PREFILL_PROVENANCE", self.resident_provenance()):
            result = WORK.account_events([self.v14_ready(), self.v13_prefill([0], [4096])])
        work = result["details"]["prefill_resident_work"][0]
        jobs = 4 * 36 * 96
        self.assertEqual(work["selected_chunk_lengths"], [512] * 8)
        self.assertEqual(work["qk_native_read_bytes"], jobs * 4096 * 1024)
        self.assertEqual(work["v_native_read_bytes"], jobs * 4096 * 64)
        self.assertEqual(work["scalar_native_read_bytes"], jobs * 128 * 8192)
        self.assertEqual(work["initial_state_read_bytes"], jobs * 8 * 8192)
        self.assertEqual(work["final_state_write_bytes"], jobs * 8 * 8192)
        self.assertEqual(work["output_write_bytes"], jobs * 128 * 2048)
        self.assertEqual(work["scalar_cache_l1_bytes_per_layer_asic"], 96 * 8192)
        self.assertEqual(WORK.resident_prefill_work([dict(chunk_length=64)])["qk_native_read_bytes"], 0)

    def test_v14_rejects_missing_geometry_scope_provenance_and_legacy_label(self):
        mutations = [
            lambda r: r["prefill_view"].pop("gdn_prefill_resident"),
            lambda r: r["prefill_view"]["gdn_prefill_resident"].pop(),
            lambda r: r["prefill_view"]["gdn_prefill_resident"][0].update(owner_enabled=True),
            lambda r: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"].pop(),
            lambda r: r["gdn_prefill_resident_enabled"]["owner"][0].update(enabled=True),
            lambda r: r["gdn_prefill_resident_enabled"]["decode"]["4"][0].update(enabled=True),
            lambda r: r.update(execution_version=WORK.HYBRID1024_VERSION),
            lambda r: r.update(execution_version=WORK.NATIVE_CONV_VERSION),
        ]
        signature = self.v14_ready()["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0]
        mutations.extend(
            lambda r, key=key: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0].pop(key)
            for key in signature
        )
        mutations.extend(
            [
                lambda r: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0].update(workers=48),
                lambda r: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0].update(dst_sync="half"),
                lambda r: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0].update(
                    scalar_cache_bytes_per_worker=4096
                ),
                lambda r: r["prefill_view"]["gdn_prefill_resident"][0]["signatures"][0]["provenance"].update(
                    reader_sha256="0" * 64
                ),
            ]
        )
        with patch.object(WORK, "RESIDENT_PREFILL_PROVENANCE", self.resident_provenance()):
            for index, mutate in enumerate(mutations):
                ready = self.v14_ready()
                mutate(ready)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    WORK.account_events([ready, self.v13_prefill([0], [1024])])
        with patch.object(WORK, "RESIDENT_PREFILL_PROVENANCE", None), self.assertRaisesRegex(
            ValueError, "not qualified"
        ):
            WORK.account_events([self.v14_ready(), self.v13_prefill([0], [1024])])

    def v13_ready(self):
        ready = self.v12_ready()
        ready["execution_version"] = WORK.HYBRID1024_VERSION
        ready["prefill_view"]["buckets"].append(1024)
        ready["prefill_view"].update(terminal_norm_1024_compact=False, terminal_norm_1024_unpacked=True)
        ready["prefill_view"]["hybrid1024"] = [
            dict(
                layer=i,
                enabled=True,
                owner_enabled=False,
                signatures=[
                    dict(
                        sequence=1024,
                        non_moe_chunks=[512, 512],
                        moe_group_rows=1024,
                        capacity_rows=1024,
                        non_moe_l1=[True, True],
                        moe_l1=False,
                        attention_chunk_size=512,
                        retained_piece_memory_configs=[
                            dict.fromkeys(
                                ("value", "mixed", "inject"),
                                "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM)",
                            )
                            for _ in range(2)
                        ],
                        retained_piece_shapes=[
                            dict(value=[1, 1, 512, 2560], mixed=[1, 1, 512, 2560], inject=[1, 1, 512, 4])
                            for _ in range(2)
                        ],
                    )
                ],
            )
            for i in range(48)
        ]
        return ready

    @staticmethod
    def v13_prefill(starts, lengths):
        chunks = WORK.serial_view_chunks(starts, lengths, buckets=WORK.VIEW_BUCKETS + [256, 512, 1024])
        for chunk in chunks:
            if chunk["chunk_length"] == 1024:
                chunk.update(
                    moe_group_rows=1024,
                    lm_head_invocations=1,
                    non_moe_chunks=[
                        dict(
                            chunk_length=512,
                            batch=1,
                            active=[True],
                            start_positions=[chunk["start_positions"][0] + offset],
                        )
                        for offset in (0, 512)
                    ],
                )
        return dict(
            phase="prefill", physical_batch=32, positions=starts, suffix_lengths=lengths, physical_chunks=chunks
        )

    def test_v13_hybrid4096_exact_fragments_group_head_and_work(self):
        step = self.v13_prefill([71], [4096])
        old_step = dict(
            step, physical_chunks=WORK.serial_view_chunks([71], [4096], buckets=WORK.VIEW_BUCKETS + [256, 512])
        )
        old = WORK.account_events([self.v12_ready(), old_step])
        new = WORK.account_events([self.v13_ready(), step])
        row = new["details"]["event_work"][0]
        old_row = old["details"]["event_work"][0]
        head = 2 * 2560 * 248320
        self.assertEqual(old["prefill"]["flops"] - new["prefill"]["flops"], 4 * head)
        for key, value in old_row["useful_matrix_flops_by_category"].items():
            self.assertEqual(row["useful_matrix_flops_by_category"][key], value - (4 * head if key == "lm_head" else 0))
        self.assertEqual(row["indexer_columns"], old_row["indexer_columns"])
        self.assertEqual(row["physical_tokens"], 4096)
        self.assertEqual(row["prefill_moe_groups"]["packed_calls_per_layer"], 4)
        self.assertEqual(row["prefill_moe_groups"]["tokens_per_layer"], 4096)
        self.assertEqual([c["capacity_rows"] for c in row["prefill_moe_groups"]["chunks"]], [[1024]] * 4)
        self.assertEqual(row["prefill_qk_rm_work"], old_row["prefill_qk_rm_work"])
        self.assertEqual(row["prefill_norm_work"]["norm_calls_per_compact_chunk"], 96)
        self.assertEqual(row["prefill_norm_work"]["terminal_compact_norm_calls"], 0)
        self.assertEqual(row["prefill_hybrid1024_work"]["lm_head_invocations"], 4)

    def test_v13_small_tail_rows_and_decode_unchanged(self):
        for length in (1, 128, 256, 511, 512, 513, 1023, 1025, 1793):
            with self.subTest(length=length):
                step = self.v13_prefill([0, 9000], [length, 3])
                old_step = dict(
                    step,
                    physical_chunks=WORK.serial_view_chunks(
                        [0, 9000], [length, 3], buckets=WORK.VIEW_BUCKETS + [256, 512]
                    ),
                )
                old = WORK.account_events([self.v12_ready(), old_step])
                new = WORK.account_events([self.v13_ready(), step])
                removed_heads = sum(c["chunk_length"] == 1024 for c in step["physical_chunks"])
                self.assertEqual(old["prefill"]["flops"] - new["prefill"]["flops"], removed_heads * 2 * 2560 * 248320)
                self.assertEqual(new["details"]["event_work"][0]["physical_tokens"], length + 3)
        for active, batch in ((1, 4), (5, 8), (9, 16), (17, 32)):
            with self.subTest(batch=batch):
                step = self.v5_step(active, batch)
                old = WORK.account_events([self.v12_ready(), step])
                new = WORK.account_events([self.v13_ready(), step])
                self.assertEqual(new["decode"], old["decode"])
                self.assertEqual(
                    new["details"]["event_work"][0]["decode_dram_bytes_by_category"],
                    old["details"]["event_work"][0]["decode_dram_bytes_by_category"],
                )

    def test_v13_rejects_missing_mismatched_or_historical_hybrid(self):
        mutations = [
            lambda r: r["prefill_view"].pop("hybrid1024"),
            lambda r: r["prefill_view"].pop("terminal_norm_1024_unpacked"),
            lambda r: r["prefill_view"].update(terminal_norm_1024_compact=True),
            lambda r: r["prefill_view"]["hybrid1024"].pop(),
            lambda r: r["prefill_view"]["hybrid1024"][0].update(owner_enabled=True),
            lambda r: r["prefill_view"]["hybrid1024"][0].update(signatures=[]),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0].update(non_moe_chunks=[1024]),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0].update(capacity_rows=512),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0].update(moe_l1=True),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0].pop("retained_piece_memory_configs"),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0]["retained_piece_memory_configs"].pop(),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0]["retained_piece_memory_configs"][0].pop(
                "inject"
            ),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0]["retained_piece_memory_configs"][1].update(
                mixed="MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1)"
            ),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0].pop("retained_piece_shapes"),
            lambda r: r["prefill_view"]["hybrid1024"][0]["signatures"][0]["retained_piece_shapes"][0].update(
                value=[1, 1, 1024, 2560]
            ),
            lambda r: r["qsa_batched_selection_geometry"]["decode"]["8"].pop(),
            lambda r: r.update(execution_version=WORK.BATCHED_QSA_VERSION),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(ready_mutation=index):
                ready = self.v13_ready()
                mutate(ready)
                with self.assertRaises(ValueError):
                    WORK.account_events([ready, self.v13_prefill([0], [1024])])
        for key in ("non_moe_chunks", "moe_group_rows", "lm_head_invocations"):
            with self.subTest(missing=key):
                step = self.v13_prefill([0], [1024])
                step["physical_chunks"][0].pop(key)
                with self.assertRaises(ValueError):
                    WORK.account_events([self.v13_ready(), step])
        step = self.v13_prefill([0], [1024])
        step["physical_chunks"][0]["non_moe_chunks"][1]["start_positions"] = [511]
        with self.assertRaises(ValueError):
            WORK.account_events([self.v13_ready(), step])

    def v12_ready(self):
        ready = self.v11_ready()
        ready["execution_version"] = WORK.BATCHED_QSA_VERSION
        indices = list(range(3, 48, 4))
        flags = lambda enabled: [dict(layer=i, enabled=enabled) for i in indices]
        ready["qsa_batched_selection"] = dict(
            owner=flags(True), prefill=flags(False), decode={str(b): flags(True) for b in (4, 8, 16)}
        )
        memory = "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM)"

        def geometries(batch):
            rows = []
            for index in indices:
                row = dict(
                    layer=index, batch=batch, sequence=1, k=512, workers=110, metadata_mode="per_row_one_row_per_core"
                )
                for stem, dtype, shape in (
                    ("score", "DataType.BFLOAT16", [batch, 1, 1, 65536]),
                    ("bound", "DataType.UINT32", [batch, 1]),
                    ("output", "DataType.UINT32", [batch, 1, 1, 512]),
                ):
                    row.update(
                        {
                            stem + "_shape": shape,
                            stem + "_padded_shape": shape,
                            stem + "_dtype": dtype,
                            stem + "_layout": "Layout.ROW_MAJOR",
                            stem + "_memory_config": memory,
                        }
                    )
                rows.append(row)
            return rows

        ready["qsa_batched_selection_geometry"] = dict(
            owner=geometries(32), prefill=[], decode={str(b): geometries(b) for b in (4, 8, 16)}
        )
        ready["qsa_batched_topk_backend"] = {
            name: dict(path="/fixture/" + name, sha256=digest, mapped_inode=123, file_inode=123)
            for name, digest in WORK.QSA_TOPK_NATIVE_SHA256.items()
        }
        return ready

    def test_v12_retained_actual_tensor_geometry(self):
        fixture = json.loads((SOURCE.parents[1] / "tests/fixtures/benchmark_qsa_selection_geometry.json").read_text())
        ready = self.v12_ready()
        geometry = ready["qsa_batched_selection_geometry"]
        # Synthetic all12 envelope validates real isolated-layer geometry only;
        # the future full48 ready collector must observe every real layer.
        for batch, actual in fixture["observed"].items():
            rows = [dict(layer=i, **actual) for i in range(3, 48, 4)]
            if batch == "32":
                geometry["owner"] = rows
            else:
                geometry["decode"][batch] = rows
        WORK.validate_batched_qsa_ready(ready)

    def test_v12_replaces_old_score_allowance_once_preserves_prefill(self):
        length = 4096
        prefill = dict(
            phase="prefill",
            physical_batch=32,
            positions=[0],
            suffix_lengths=[length],
            physical_chunks=WORK.serial_view_chunks([0], [length], buckets=WORK.VIEW_BUCKETS + [256, 512]),
        )
        for active, batch in ((1, 4), (5, 8), (9, 16), (17, 32), (32, 32)):
            with self.subTest(active=active, batch=batch):
                step = self.v5_step(active, batch)
                old = WORK.account_events([self.v11_ready(), prefill, step])
                new = WORK.account_events([self.v12_ready(), prefill, step])
                self.assertEqual(new["prefill"], old["prefill"])
                old_row = old["details"]["event_work"][1]
                row = new["details"]["event_work"][1]
                old_categories = old_row["decode_dram_bytes_by_category"]
                new_categories = row["decode_dram_bytes_by_category"]
                self.assertNotIn("qsa_indexer_score_write_read", new_categories)
                payload = WORK.batched_qsa_selection_payload(step["decode_execution_positions"])
                for key, value in payload.items():
                    self.assertEqual(new_categories[key], value)
                self.assertEqual(
                    new["decode"]["dram_bytes"],
                    old["decode"]["dram_bytes"]
                    - old_categories["qsa_indexer_score_write_read"]
                    + sum(payload.values()),
                )
                self.assertEqual(row["qsa_batched_selection_work"]["active_topk_workers"], batch)
                self.assertEqual(row["useful_matrix_flops"], old_row["useful_matrix_flops"])
        with self.assertRaisesRegex(ValueError, "historical"):
            WORK.account_events(
                [dict(self.v12_ready(), execution_version=WORK.PREFILL_QK_RM_VERSION), self.v5_step(4, 4)]
            )

    def test_v12_mask_boundary_and_inactive_worker_metadata(self):
        positions = [0, 1022, 1023, 262143]
        payload = WORK.batched_qsa_selection_payload(positions)
        factor = 4 * 12
        # Complete columns0/255/256/65536: partial block only255.
        self.assertEqual(payload["qsa_mask_partial_score_reads"], factor * 512)
        self.assertEqual(payload["qsa_mask_suffix_score_writes"], factor * 2 * (65536 + 65536 + 65536 - 256))
        self.assertEqual(payload["qsa_topk_score_prefix_reads"], factor * 2 * (512 * 3 + 65536))
        self.assertEqual(payload["qsa_topk_metadata_reads"], factor * 110 * 4)
        self.assertEqual(payload["qsa_mask_positions_reads"], factor * (4 * 110 + 4) * 128)
        self.assertEqual(payload["qsa_per_row_bound_writes"], factor * 4 * 32)
        self.assertEqual(payload["qsa_batched_score_concat_read_write"], factor * 2 * 4 * 65536 * 2)
        for positions in ([0], [-1] * 4, [262144] * 4):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                WORK.batched_qsa_selection_payload(positions)

    def test_v12_rejects_partial_geometry_wrong_native_and_scope(self):
        mutations = [
            lambda r: r["qsa_batched_selection"]["prefill"][0].update(enabled=True),
            lambda r: r["qsa_batched_selection_geometry"]["owner"].pop(),
            lambda r: r["qsa_batched_selection_geometry"]["decode"].pop("8"),
            lambda r: r["qsa_batched_topk_backend"].pop("_ttnn.so"),
            lambda r: r["qsa_batched_topk_backend"]["_ttnn.so"].update(sha256="wrong"),
            lambda r: r["qsa_batched_topk_backend"]["_ttnn.so"].update(file_inode=999),
        ]
        for field, value in (
            ("metadata_mode", "scalar_max"),
            ("workers", 120),
            ("bound_shape", [1]),
            ("score_dtype", "DataType.FLOAT32"),
            ("output_layout", "Layout.TILE"),
            ("score_memory_config", "L1"),
        ):
            mutations.append(
                lambda r, f=field, v=value: r["qsa_batched_selection_geometry"]["decode"]["4"][0].update({f: v})
            )
        for index, mutate in enumerate(mutations):
            ready = self.v12_ready()
            mutate(ready)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                WORK.account_events([ready, self.v5_step(1, 4)])

    def v11_ready(self):
        ready = self.v10_ready()
        ready["execution_version"] = WORK.PREFILL_QK_RM_VERSION
        indices = [i for i in range(48) if i % 4 != 3]
        flags = lambda enabled: [dict(layer=i, enabled=enabled) for i in indices]
        ready["gdn_prefill_qk_rm_enabled"] = dict(
            owner=flags(False), prefill=flags(True), decode={str(b): flags(False) for b in (4, 8, 16)}
        )
        memory = "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM)"
        signatures = []
        for seq in (128, 256, 512):
            geometry = dict(
                logical_shape=[1, seq, 4, 128],
                tile_padded_shape=[1, seq, 32, 128],
                rm_padded_shape=[1, seq, 4, 128],
                tile_dtype="DataType.FLOAT32",
                rm_dtype="DataType.FLOAT32",
                tile_memory_config=memory,
                rm_memory_config=memory,
                tile_layout="Layout.TILE",
                rm_layout="Layout.ROW_MAJOR",
            )
            signatures.append(dict(sequence=seq, q=copy.deepcopy(geometry), k=copy.deepcopy(geometry)))
        ready["prefill_view"]["gdn_prefill_qk_rm"] = [
            dict(layer=i, enabled=True, owner_enabled=False, signatures=copy.deepcopy(signatures)) for i in indices
        ]
        return ready

    def test_v11_prefill_qk_rm_preserves_v10_headline_work(self):
        length = 4096 + 256 + 128 + 64 + 1
        chunks = WORK.serial_view_chunks([0], [length], buckets=WORK.VIEW_BUCKETS + [256, 512])
        event = dict(phase="prefill", physical_batch=32, positions=[0], suffix_lengths=[length], physical_chunks=chunks)
        for batch in (4, 8, 16, 32):
            with self.subTest(batch=batch):
                events = [event, self.v5_step(batch, batch)]
                old = WORK.account_events([self.v10_ready(), *events])
                new = WORK.account_events([self.v11_ready(), *events])
                self.assertEqual(new["prefill"], old["prefill"])
                self.assertEqual(new["decode"], old["decode"])
                detail = new["details"]["event_work"][0]["prefill_qk_rm_work"]
                self.assertEqual(detail["selected_chunk_lengths"], [512] * 8 + [256, 128])
                tokens = 4096 + 256 + 128
                self.assertEqual(detail["qk_native_read_bytes"], 4 * 36 * tokens * 48 * 1024)
                self.assertEqual(detail["historical_qk_native_read_bytes"], 4 * 36 * tokens * 48 * 32768)
                self.assertEqual(detail["added_tile_to_rm_read_write_bytes"], 4 * 36 * tokens * 36864)
                self.assertEqual(
                    new["details"]["event_work"][0]["prefill_norm_work"],
                    old["details"]["event_work"][0]["prefill_norm_work"],
                )
        with self.assertRaisesRegex(ValueError, "historical"):
            WORK.account_events([dict(self.v11_ready(), execution_version=WORK.NORM_COMPACT_VERSION), event])

    def test_v11_prefill_qk_rm_rejects_missing_or_changed_actual_geometry(self):
        mutations = [
            lambda r: r["prefill_view"]["gdn_prefill_qk_rm"].pop(),
            lambda r: r["prefill_view"]["gdn_prefill_qk_rm"][0]["signatures"].pop(),
            lambda r: r["prefill_view"]["gdn_prefill_qk_rm"][0].update(owner_enabled=True),
            lambda r: r["gdn_prefill_qk_rm_enabled"]["decode"]["8"][0].update(enabled=True),
            lambda r: r["gdn_prefill_qk_rm_enabled"]["owner"][0].update(enabled=True),
        ]
        for field, value in (
            ("logical_shape", [1, 128, 32, 128]),
            ("tile_padded_shape", [1, 128, 4, 128]),
            ("rm_padded_shape", [1, 128, 32, 128]),
            ("rm_dtype", "DataType.BFLOAT16"),
            ("rm_memory_config", "L1"),
            ("tile_memory_config", "unknown"),
            ("rm_layout", "Layout.TILE"),
        ):
            mutations.append(
                lambda r, f=field, v=value: r["prefill_view"]["gdn_prefill_qk_rm"][0]["signatures"][0]["q"].update(
                    {f: v}
                )
            )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                ready = self.v11_ready()
                mutate(ready)
                with self.assertRaises(ValueError):
                    WORK.account_events([ready, self.v5_step(8, 8)])

    def v10_ready(self):
        ready = self.v9_ready()
        ready["execution_version"] = WORK.NORM_COMPACT_VERSION
        memory = "MemoryConfig(memory_layout=TensorMemoryLayout.INTERLEAVED, buffer_type=BufferType.DRAM)"
        signatures = [
            dict(
                sequence=seq,
                input_shape=[1, seq, 4, 640],
                input_padded_shape=[1, seq, 32, 640],
                compact_shape=[1, 1, 4 * seq, 640],
                compact_padded_shape=[1, 1, 4 * seq, 640],
                restored_shape=[1, seq, 4, 640],
                input_memory_config=memory,
                compact_memory_config=memory,
                restored_memory_config=memory,
                input_dtype="DataType.BFLOAT16",
                statistics_dtype="DataType.FLOAT32",
                weight_shape=[1, 1, 4, 640],
                weight_dtype="DataType.BFLOAT16",
                compute_kernel_config=dict(
                    math_fidelity="MathFidelity.HiFi4",
                    math_approx_mode="False",
                    fp32_dest_acc_en="True",
                    packer_l1_acc="True",
                    dst_full_sync_en="False",
                ),
                restore_before_stream_weight=True,
            )
            for seq in (128, 256, 512)
        ]
        ready["prefill_view"]["norm_compaction"] = dict(
            layers=[dict(layer=i, signatures=copy.deepcopy(signatures)) for i in range(48)],
            terminal_signatures=copy.deepcopy(signatures),
        )
        return ready

    def test_v10_retained_real_norm_signature_memory_spelling(self):
        fixture = json.loads((SOURCE.parents[1] / "tests/fixtures/benchmark_norm_signatures.json").read_text())
        ready = self.v10_ready()
        signatures = fixture["observed_signatures"]
        ready["prefill_view"]["norm_compaction"] = dict(
            layers=[dict(layer=i, signatures=copy.deepcopy(signatures)) for i in range(48)],
            terminal_signatures=copy.deepcopy(signatures),
        )
        # Synthetic complete envelope tests validation; retained signatures came
        # from a reduced-layer probe, not a claim of full48 warmup.
        WORK.validate_prefill_norm_ready(ready)
        ready["prefill_view"]["norm_compaction"]["terminal_signatures"][0]["compute_kernel_config"][
            "dst_full_sync_en"
        ] = "True"
        with self.assertRaises(ValueError):
            WORK.validate_prefill_norm_ready(ready)

    def test_v10_norm_metadata_preserves_prefill_flops_decode_bytes(self):
        chunks = WORK.serial_view_chunks([0], [4096 + 128 + 64], buckets=WORK.VIEW_BUCKETS + [256, 512])
        prefill = dict(
            phase="prefill", physical_batch=32, positions=[0], suffix_lengths=[4096 + 128 + 64], physical_chunks=chunks
        )
        events = [prefill, self.v5_step(32, 32)]
        new = WORK.account_events([self.v10_ready(), *events])
        old = WORK.account_events([self.v9_ready(), *events])
        self.assertEqual(new["prefill"], old["prefill"])
        self.assertEqual(new["decode"], old["decode"])
        work = new["details"]["event_work"][0]["prefill_norm_work"]
        self.assertEqual(work["compact_chunk_lengths"], [512] * 8 + [128])
        self.assertEqual(work["norm_calls_per_compact_chunk"], 97)
        with self.assertRaises(ValueError):
            WORK.account_events([dict(self.v10_ready(), execution_version=WORK.HISTORY_CONV_VERSION), *events])

    def test_v10_norm_geometry_precision_and_terminal_fail_closed(self):
        for field, value in (
            ("input_padded_shape", [1, 128, 4, 640]),
            ("compact_shape", [1, 1, 128, 2560]),
            ("statistics_dtype", "DataType.BFLOAT16"),
            ("restore_before_stream_weight", False),
            ("weight_shape", [1, 1, 1, 640]),
            ("restored_memory_config", "missing"),
            ("compute_kernel_config", {}),
        ):
            ready = self.v10_ready()
            ready["prefill_view"]["norm_compaction"]["layers"][0]["signatures"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                WORK.account_events([ready, self.v5_step(8, 8)])
        ready = self.v10_ready()
        ready["prefill_view"]["norm_compaction"]["terminal_signatures"].pop()
        with self.assertRaises(ValueError):
            WORK.account_events([ready, self.v5_step(8, 8)])

    def v9_ready(self):
        ready = self.v8_ready()
        ready["execution_version"] = WORK.HISTORY_CONV_VERSION
        flags = lambda enabled: [dict(layer=i, enabled=enabled) for i in range(48) if i % 4 != 3]
        ready["gdn_conv_batch_packed"]["owner"] = flags(False)
        ready["gdn_conv_history_packed"] = dict(
            owner=flags(True), prefill=flags(False), decode={str(b): flags(b in (8, 16)) for b in (4, 8, 16)}
        )
        geometries = lambda batch: [dict(layer=i, **WORK.history_conv_geometry(batch)) for i in range(48) if i % 4 != 3]
        ready["gdn_conv_history_geometry"] = dict(
            owner=geometries(32), prefill=[], decode={str(b): geometries(b) for b in (8, 16)}
        )
        return ready

    def test_v9_history_sequence_actual_native_work_and_historical_ledger(self):
        for batch in (4, 8, 16, 32):
            event = self.v5_step(batch, batch)
            current = WORK.account_events([self.v9_ready(), event])
            old = WORK.account_events([self.v8_ready(), event])
            baseline = WORK.gdn_convolution_payload(batch, packed=batch == 32)
            new = WORK.gdn_history_convolution_payload(batch) if batch > 4 else baseline
            self.assertEqual(
                current["decode"]["dram_bytes"] - old["decode"]["dram_bytes"],
                sum(new.values()) - sum(baseline.values()),
            )
            if batch > 4:
                mt = batch // 8
                self.assertEqual(new["gdn_conv_native_input_reads"], 4 * 36 * (32 * mt + 3 * (mt - 1)) * 2560 * 2)
                self.assertEqual(new["gdn_conv_native_history_reads"], 4 * 36 * 3 * 2560 * 2)
                self.assertEqual(new["gdn_conv_native_tap_reads"], 4 * 36 * 4 * 80 * mt * 2048)
                self.assertEqual(new["gdn_conv_native_output_writes"], 4 * 36 * 80 * mt * 2048)
            work = current["details"]["event_work"][0]["gdn_convolution_work"]
            self.assertEqual(work["native_output_rows_per_layer"], batch * (4 if batch > 4 else 32))
        chunks = WORK.serial_view_chunks([0], [4096], buckets=WORK.VIEW_BUCKETS + [256, 512])
        prefill = dict(phase="prefill", physical_batch=32, positions=[0], suffix_lengths=[4096], physical_chunks=chunks)
        self.assertEqual(
            WORK.account_events([self.v9_ready(), prefill])["prefill"],
            WORK.account_events([self.v8_ready(), prefill])["prefill"],
        )

    def test_v9_history_flags_and_selected_rows_fail_closed(self):
        for scope in ("owner", "4", "8", "16"):
            ready = self.v9_ready()
            rows = (
                ready["gdn_conv_history_packed"]["owner"]
                if scope == "owner"
                else ready["gdn_conv_history_packed"]["decode"][scope]
            )
            rows[0]["enabled"] = not rows[0]["enabled"]
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                WORK.account_events([ready, self.v5_step(8, 8)])
        ready = self.v9_ready()
        ready["gdn_conv_history_geometry"]["decode"]["8"][0]["selected_rows"][0] = 0
        with self.assertRaises(ValueError):
            WORK.account_events([ready, self.v5_step(8, 8)])
        with self.assertRaises(ValueError):
            WORK.gdn_history_convolution_payload(4)

    def v8_ready(self):
        ready = self.v7_ready()
        ready["execution_version"] = WORK.GROUP512_VERSION
        ready["prefill_view"]["buckets"] = WORK.VIEW_BUCKETS + [256, 512]
        ready["prefill_view"]["moe_prefill_group_size"] = [
            dict(layer=i, group_size=512, capacity_rows=512) for i in range(48)
        ]
        return ready

    def test_v8_actual512_chunks_groups_and_historical_guard(self):
        ready = self.v8_ready()
        chunks = WORK.serial_view_chunks([0, 0], [4096, 257], buckets=ready["prefill_view"]["buckets"])
        event = dict(
            phase="prefill", physical_batch=32, positions=[0, 0], suffix_lengths=[4096, 257], physical_chunks=chunks
        )
        result = WORK.account_events([ready, event, self.v5_step(32, 32)])
        groups = result["details"]["event_work"][0]["prefill_moe_groups"]
        self.assertEqual(groups["tokens_per_layer"], 4353)
        self.assertEqual(groups["packed_calls_per_layer"], 9)
        self.assertEqual(groups["native_singleton_calls_per_layer"], 1)
        self.assertEqual([row["capacity_rows"] for row in groups["chunks"]], [[512]] * 8 + [[256], [32]])
        self.assertEqual(groups["useful_routed_matrix_flops"], 48 * 10 * 4353 * 3 * 2 * 2560 * 640)
        self.assertEqual(result["decode"], WORK.account_events([self.v7_ready(), self.v5_step(32, 32)])["decode"])
        for version in (WORK.COMPACT_VIEW_VERSION, WORK.PREFILL_GROUP256_VERSION, WORK.CONV_PACK_VERSION):
            with self.subTest(version=version), self.assertRaises(ValueError):
                WORK.account_events([dict(ready, execution_version=version), event])
        altered = self.v8_ready()
        altered["prefill_view"]["moe_prefill_group_size"][0]["group_size"] = 256
        with self.assertRaises(ValueError):
            WORK.account_events([altered, event])

    def v7_ready(self, batch=32):
        ready = self.v6_ready() if batch == 32 else self.cyclic_ready(1)
        ready["execution_version"] = WORK.CONV_PACK_VERSION if batch == 32 else WORK.NATIVE_CONV_VERSION
        flags = lambda enabled: [dict(layer=i, enabled=enabled) for i in range(48) if i % 4 != 3]
        ready["gdn_conv_batch_packed"] = dict(
            owner=flags(batch == 32),
            prefill=flags(False) if batch == 32 else None,
            decode={str(b): flags(False) for b in (4, 8, 16)} if batch == 32 else {},
        )
        if batch == 32:
            geometry = dict(
                batch=32,
                sequence=1,
                input_shape=[1, 32, 81920],
                history_shape=[1, 3, 81920],
                widths=[16384, 16384, 49152],
                dtype="DataType.BFLOAT16",
                layout="Layout.ROW_MAJOR",
                native_output_shapes=[[1, 32, width] for width in (16384, 16384, 49152)],
                restored_output_shapes=[[32, 1, width] for width in (512, 512, 1536)],
            )
            ready["gdn_conv_pack_geometry"] = [dict(layer=i, **geometry) for i in range(48) if i % 4 != 3]
        return ready

    def test_v7_conv_known_payload_and_historical_native_reconstruction(self):
        for batch in (1, 4, 8, 16, 32):
            with self.subTest(batch=batch):
                new_ready = self.v7_ready(1 if batch == 1 else 32)
                old_ready = self.cyclic_ready(1) if batch == 1 else self.v6_ready()
                event = (
                    dict(phase="decode", physical_batch=1, positions=[4096], suffix_lengths=[1])
                    if batch == 1
                    else self.v5_step(batch, batch)
                )
                new = WORK.account_events([new_ready, event])
                old = WORK.account_events([old_ready, event])
                added = WORK.gdn_convolution_payload(batch, packed=batch == 32)
                self.assertEqual(new["decode"]["dram_bytes"] - old["decode"]["dram_bytes"], sum(added.values()))
                self.assertEqual(added["gdn_conv_native_tap_reads"], 4 * 36 * 4 * batch * 80 * 2048)
                self.assertEqual(added["gdn_conv_native_history_reads"], 4 * 36 * 3 * batch * 2560 * 2)
                self.assertEqual(
                    added["gdn_conv_native_start_reads"], 4 * 36 * 4 * (110 if batch == 32 else 20 * batch)
                )
                self.assertEqual(
                    added["gdn_conv_native_input_reads"], 0 if batch == 1 else 4 * 36 * batch * 32 * 2560 * 2
                )
                self.assertEqual(new["prefill"], old["prefill"])
        with self.assertRaises(ValueError):
            WORK.gdn_convolution_payload(16, packed=True)

    def test_v7_actual_conv_flags_fail_closed(self):
        for batch in (1, 32):
            ready = self.v7_ready(batch)
            WORK.validate_conv_ready(ready)
            for scope in ("owner", "prefill", "decode") if batch == 32 else ("owner",):
                altered = copy.deepcopy(ready)
                rows = altered["gdn_conv_batch_packed"][scope]
                if scope == "decode":
                    rows = rows["16"]
                rows[0]["enabled"] = not rows[0]["enabled"]
                with self.subTest(batch=batch, scope=scope), self.assertRaises(ValueError):
                    WORK.validate_conv_ready(altered)

    def v6_ready(self):
        ready = self.v5_ready()
        ready["execution_version"] = WORK.PREFILL_GROUP256_VERSION
        expected = lambda size: [dict(layer=i, group_size=size, capacity_rows=size) for i in range(48)]
        ready["moe_prefill_group_size"] = expected(128)
        ready["prefill_view"]["moe_prefill_group_size"] = expected(256)
        ready["prefill_view"]["buckets"] = WORK.VIEW_BUCKETS + [256]
        ready["decode_moe_prefill_group_size"] = {str(b): expected(128) for b in (4, 8, 16)}
        return ready

    def test_v6_actual_prefill_group_calls_preserve_useful_flops_and_decode(self):
        ready = self.v6_ready()
        chunks = WORK.serial_view_chunks([0, 0], [4096, 129], buckets=ready["prefill_view"]["buckets"])
        prefill = dict(
            phase="prefill", physical_batch=32, positions=[0, 0], suffix_lengths=[4096, 129], physical_chunks=chunks
        )
        current = WORK.account_events([ready, prefill, self.v5_step(1, 4)])
        historical = WORK.account_events(
            [dict(ready, execution_version=WORK.COMPACT_VIEW_VERSION), prefill, self.v5_step(1, 4)]
        )
        self.assertEqual(current["prefill"], historical["prefill"])
        self.assertEqual(current["decode"], historical["decode"])
        grouped = current["details"]["event_work"][0]["prefill_moe_groups"]
        self.assertEqual(grouped["packed_calls_per_layer"], 17)
        self.assertEqual(grouped["native_singleton_calls_per_layer"], 1)
        self.assertEqual(grouped["tokens_per_layer"], 4225)
        self.assertEqual(grouped["total_calls_all48_layers"], 18 * 48)
        self.assertEqual(
            grouped["useful_routed_matrix_flops"],
            current["details"]["prefill_matrix_flops_by_category"]["routed_experts_48"],
        )
        self.assertEqual([row["capacity_rows"] for row in grouped["chunks"]], [[256]] * 16 + [[128], [32]])
        old = WORK.prefill_moe_group_work(chunks, group_size=128)
        self.assertEqual(old["packed_calls_per_layer"], 33)
        self.assertEqual(old["tokens_per_layer"], grouped["tokens_per_layer"])

    def test_v6_group_attestation_fails_closed(self):
        step = self.v5_step(1, 4)
        for scope in ("owner", "prefill", "decode"):
            ready = self.v6_ready()
            rows = (
                ready["moe_prefill_group_size"]
                if scope == "owner"
                else (
                    ready["prefill_view"]["moe_prefill_group_size"]
                    if scope == "prefill"
                    else ready["decode_moe_prefill_group_size"]["4"]
                )
            )
            rows[0]["group_size"] = 64
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                WORK.account_events([ready, step])

    def v5_ready(self):
        ready = self.cyclic_ready(32)
        ready["execution_version"] = WORK.COMPACT_VIEW_VERSION
        ready["layer_indices"] = list(range(48))
        ready["context"] = 262144
        ready["prefill_qsa_decode_fold_vectorized"] = [dict(layer=i, enabled=False) for i in range(3, 48, 4)]
        flags = [dict(layer=i, enabled=True) for i in range(3, 48, 4)]
        ready["qsa_decode_fold_vectorized"] = copy.deepcopy(flags)
        runtime = dict(
            enabled=True,
            buckets=[4, 8, 16],
            fallback_batch=32,
            selection="smallest_bucket_covering_nonempty_contiguous_active_prefix_else_owner",
            sampler_batch=32,
            sampler_owner="original_generator",
            position_advance="original_owner_positions_and_active_mask_once",
            views={},
        )
        for batch in (4, 8, 16):
            runtime["views"][str(batch)] = dict(
                execution_batch=batch,
                layers=copy.deepcopy(ready["prefill_view"]["layers"]),
                layer_indices=list(range(48)),
                context=262144,
                owner_public_state_shared=True,
                owner_precision_policy_shared=True,
                owner_lm_head_shared=True,
                precision=ready["prefill_view"]["precision"],
                gdn_recurrence_broadcast=copy.deepcopy(ready["gdn_recurrence_broadcast"]),
                indexer_prefix_cap=copy.deepcopy(flags),
                qsa_decode_fold_vectorized=copy.deepcopy(flags),
                pooled_gather_partitions=copy.deepcopy(ready["pooled_gather_partitions"]),
                input_bridge="device_prefix_slice_tokens_positions_slots_full_page_rows",
                logits_bridge="owner_tile_to_row_major_prefix_slice_write_tile_active_mask_merge_copy",
                input_shapes=dict(tokens=[batch, 1], positions=[batch, 32], slots=[batch, 32], pages=[batch, 8192]),
                logit_shapes=dict(view=[1, 1, batch, 62080], owner=[1, 1, 32, 62080], row_major_work=[1, 1, 32, 62080]),
            )
        ready["decode_runtime"] = runtime
        return ready

    def v5_step(self, active, batch, mask=None):
        mask = mask or [True] * active + [False] * (32 - active)
        return dict(
            phase="decode",
            physical_batch=32,
            positions=[4096] * active,
            suffix_lengths=[1] * active,
            decode_execution_batch=batch,
            decode_active_rows=mask,
            decode_execution_positions=[4096 if a else 0 for a in mask[:batch]],
            sampler_batch=32,
        )

    def test_v5_physical_bucket_work_preserves_padding_and_owner_sampling(self):
        ready = self.v5_ready()
        for active, batch in ((1, 4), (4, 4), (5, 8), (9, 16), (17, 32), (32, 32)):
            with self.subTest(active=active, batch=batch):
                result = WORK.account_events([ready, self.v5_step(active, batch)])
                row = result["details"]["event_work"][0]
                self.assertEqual(row["batch"], 32)
                self.assertEqual(row["execution_batches"], [batch])
                self.assertEqual(row["physical_tokens"], batch)
                self.assertEqual(row["active_tokens"], active)
                self.assertEqual(row["decode_routing_bounds"]["nonzero_routes_per_layer"]["upper"], batch * 10)
                traffic = row["decode_dram_bytes_by_category"]
                self.assertEqual(traffic["gdn_recurrence_native_operand_reads"], 4 * 36 * batch * 48 * 11 * 4096)
                self.assertEqual(traffic["head_logits_write_and_sampler_read"], 2 * 32 * 248320 * 2)
                self.assertEqual(traffic["qsa_pooled_gather_positions_read"], 4 * 12 * batch * 110 * 128)
                self.assertEqual(traffic["qsa_pooled_gather_page_table_read"], 4 * 12 * active * 110 * 544)
        a = WORK.account_events([ready, self.v5_step(1, 4)])["details"]["event_work"][0]
        b = WORK.account_events([ready, self.v5_step(4, 4)])["details"]["event_work"][0]
        self.assertEqual(a["matrix_flops_by_category"]["gdn_packed_36"], b["matrix_flops_by_category"]["gdn_packed_36"])
        self.assertLess(a["useful_matrix_flops"], b["useful_matrix_flops"])

    def test_v5_prefill_useful_work_remains_v4(self):
        step = dict(
            phase="prefill",
            physical_batch=32,
            positions=[0, 7],
            suffix_lengths=[4096, 129],
            physical_chunks=WORK.serial_view_chunks([0, 7], [4096, 129]),
        )
        before = WORK.account_events([self.cyclic_ready(32), step])
        after = WORK.account_events([self.v5_ready(), step])
        self.assertEqual(before["prefill"], after["prefill"])
        self.assertEqual(after["details"]["event_work"][0]["physical_tokens"], 4225)

    def test_v5_nonprefix_fallback_and_actual_view_partitions(self):
        ready = self.v5_ready()
        mask = [False, True] + [False] * 30
        result = WORK.account_events([ready, self.v5_step(1, 32, mask)])
        self.assertEqual(result["details"]["event_work"][0]["execution_batches"], [32])
        with self.assertRaisesRegex(ValueError, "selection"):
            WORK.account_events([ready, self.v5_step(1, 4, mask)])
        for part in ready["decode_runtime"]["views"]["4"]["pooled_gather_partitions"]:
            part["workers"] = 100
        traffic = WORK.account_events([ready, self.v5_step(1, 4)])["details"]["decode_dram_bytes_by_category"]
        self.assertEqual(traffic["qsa_pooled_gather_positions_read"], 4 * 12 * 4 * 100 * 128)

    def test_v5_missing_or_conflicting_geometry_fails_closed(self):
        for key in ("decode_execution_batch", "decode_active_rows", "decode_execution_positions"):
            step = self.v5_step(1, 4)
            del step[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                WORK.account_events([self.v5_ready(), step])
        for key in (
            "precision",
            "layers",
            "indexer_prefix_cap",
            "qsa_decode_fold_vectorized",
            "pooled_gather_partitions",
            "owner_lm_head_shared",
            "input_shapes",
            "logit_shapes",
        ):
            ready = self.v5_ready()
            del ready["decode_runtime"]["views"]["4"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                WORK.account_events([ready, self.v5_step(1, 4)])
        step = self.v5_step(1, 4)
        step["decode_execution_positions"][-1] = 12
        with self.assertRaisesRegex(ValueError, "padding"):
            WORK.account_events([self.v5_ready(), step])

    def test_v5_ready_version_requires_observed_owner_and_view_paths(self):
        from types import SimpleNamespace

        from models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_phase import execution_metadata

        ready = self.v5_ready()
        owner_layers = [
            SimpleNamespace(
                kind="indexed_attention" if i % 4 == 3 else "linear_attention",
                attention=SimpleNamespace(
                    recurrence_broadcast=True,
                    _decode_fold_vectorized=True,
                    _pooled_gather_partition=dict(kind="cyclic", work=8192, workers=110),
                ),
            )
            for i in range(48)
        ]
        prefill_layers = [
            SimpleNamespace(kind=layer.kind, attention=SimpleNamespace(_decode_fold_vectorized=False))
            for layer in owner_layers
        ]
        model = SimpleNamespace(
            batch_size=32,
            generator=SimpleNamespace(
                model=SimpleNamespace(layer_indices=list(range(48)), layers=owner_layers),
                prefill_view=SimpleNamespace(
                    model=SimpleNamespace(layer_indices=list(range(48)), layers=prefill_layers)
                ),
                decode_runtime_summary=lambda: ready["decode_runtime"],
            ),
        )
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.COMPACT_VIEW_VERSION
        )
        model.generator.decode_views = {
            batch: SimpleNamespace(
                model=SimpleNamespace(layer_indices=list(range(48)), layers=copy.deepcopy(owner_layers))
            )
            for batch in (4, 8, 16)
        }
        ready["prefill_view"]["moe_prefill_group_size"] = self.v6_ready()["prefill_view"]["moe_prefill_group_size"]
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.PREFILL_GROUP256_VERSION
        )
        for layer in owner_layers:
            if layer.kind == "linear_attention":
                layer.attention._decode_conv_batch_packed = True
                layer.attention._decode_conv_pack_geometry = {
                    k: v for k, v in self.v7_ready()["gdn_conv_pack_geometry"][0].items() if k != "layer"
                }
        for native in [
            model.generator.prefill_view.model,
            *[view.model for view in model.generator.decode_views.values()],
        ]:
            for layer in native.layers:
                if layer.kind == "linear_attention":
                    layer.attention._decode_conv_batch_packed = False
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.CONV_PACK_VERSION
        )
        ready["prefill_view"]["moe_prefill_group_size"] = self.v8_ready()["prefill_view"]["moe_prefill_group_size"]
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.GROUP512_VERSION
        )
        natives = [
            (32, model.generator.model),
            (1, model.generator.prefill_view.model),
            *[(b, view.model) for b, view in model.generator.decode_views.items()],
        ]
        for batch, native in natives:
            for layer in native.layers:
                if layer.kind == "linear_attention":
                    layer.attention._decode_conv_batch_packed = False
                    layer.attention._decode_conv_history_packed = batch in (8, 16, 32)
                    if batch in (8, 16, 32):
                        layer.attention._decode_conv_history_geometry = WORK.history_conv_geometry(batch)
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.HISTORY_CONV_VERSION
        )
        ready["prefill_view"]["norm_compaction"] = self.v10_ready()["prefill_view"]["norm_compaction"]
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.NORM_COMPACT_VERSION
        )
        ready["prefill_view"]["gdn_prefill_qk_rm"] = self.v11_ready()["prefill_view"]["gdn_prefill_qk_rm"]
        for layer in model.generator.prefill_view.model.layers:
            if layer.kind == "linear_attention":
                layer.attention.prefill_qk_rm_enabled = True
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.PREFILL_QK_RM_VERSION
        )
        from unittest.mock import patch

        for batch, native in natives:
            if batch == 1:
                continue
            rows = (
                self.v12_ready()["qsa_batched_selection_geometry"]["owner"]
                if batch == 32
                else self.v12_ready()["qsa_batched_selection_geometry"]["decode"][str(batch)]
            )
            for index, layer in zip(native.layer_indices, native.layers):
                if layer.kind != "linear_attention":
                    layer.attention._qsa_batched_selection = True
                    layer.attention._qsa_batched_selection_geometry = {
                        k: v for k, v in next(r for r in rows if r["layer"] == index).items() if k != "layer"
                    }
        with patch(
            "models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_phase.loaded_topk_backend",
            return_value=self.v12_ready()["qsa_batched_topk_backend"],
        ):
            self.assertEqual(
                execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.BATCHED_QSA_VERSION
            )
            ready["prefill_view"]["buckets"] = self.v13_ready()["prefill_view"]["buckets"]
            ready["prefill_view"]["hybrid1024"] = self.v13_ready()["prefill_view"]["hybrid1024"]
            ready["prefill_view"].update(terminal_norm_1024_compact=False, terminal_norm_1024_unpacked=True)
            self.assertEqual(
                execution_metadata(model, ready["prefill_view"], True)["execution_version"], WORK.HYBRID1024_VERSION
            )
            ready["prefill_view"]["gdn_prefill_resident"] = self.v14_ready()["prefill_view"]["gdn_prefill_resident"]
            for layer in prefill_layers:
                if layer.kind == "linear_attention":
                    layer.attention.prefill_resident_enabled = True
            with patch(
                "models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_work.RESIDENT_PREFILL_PROVENANCE",
                self.resident_provenance(),
            ):
                metadata = execution_metadata(model, ready["prefill_view"], True)
                self.assertEqual(metadata["execution_version"], WORK.RESIDENT_PREFILL_VERSION)
                self.assertEqual(
                    metadata["gdn_prefill_resident_enabled"], self.v14_ready()["gdn_prefill_resident_enabled"]
                )
                owner_layers[0].attention.prefill_resident_enabled = True
                self.assertEqual(
                    execution_metadata(model, ready["prefill_view"], True)["execution_version"],
                    "unqualified_resident_prefill",
                )
                del owner_layers[0].attention.prefill_resident_enabled
                ready["prefill_view"]["gdn_prefill_resident"][0]["signatures"].pop()
                self.assertEqual(
                    execution_metadata(model, ready["prefill_view"], True)["execution_version"],
                    "unqualified_resident_prefill",
                )
            del ready["prefill_view"]["gdn_prefill_resident"]
            for layer in prefill_layers:
                if layer.kind == "linear_attention":
                    del layer.attention.prefill_resident_enabled
            ready["prefill_view"]["hybrid1024"][0]["signatures"].clear()
            self.assertEqual(
                execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_hybrid1024"
            )
            del ready["prefill_view"]["hybrid1024"]
            ready["prefill_view"]["buckets"] = self.v12_ready()["prefill_view"]["buckets"]
        with patch(
            "models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_phase.loaded_topk_backend",
            side_effect=ValueError("unqualified native"),
        ):
            self.assertEqual(
                execution_metadata(model, ready["prefill_view"], True)["execution_version"],
                "unqualified_batched_qsa_selection",
            )
        for _, native in natives:
            for layer in native.layers:
                if layer.kind != "linear_attention" and hasattr(layer.attention, "_qsa_batched_selection"):
                    del layer.attention._qsa_batched_selection
                    del layer.attention._qsa_batched_selection_geometry
        model.generator.decode_views[8].model.layers[0].attention.prefill_qk_rm_enabled = True
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_prefill_qk_rm"
        )
        del model.generator.decode_views[8].model.layers[0].attention.prefill_qk_rm_enabled
        ready["prefill_view"]["gdn_prefill_qk_rm"][0]["signatures"].pop()
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_prefill_qk_rm"
        )
        del ready["prefill_view"]["gdn_prefill_qk_rm"]
        for layer in model.generator.prefill_view.model.layers:
            if layer.kind == "linear_attention":
                del layer.attention.prefill_qk_rm_enabled
        ready["prefill_view"]["norm_compaction"]["terminal_signatures"].pop()
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"],
            "unqualified_prefill_norm_compaction",
        )
        del ready["prefill_view"]["norm_compaction"]
        model.generator.decode_views[4].model.layers[0].attention._decode_conv_history_packed = True
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_history_conv"
        )
        for batch, native in natives:
            for layer in native.layers:
                if layer.kind == "linear_attention":
                    del layer.attention._decode_conv_history_packed
                    layer.attention._decode_conv_batch_packed = batch == 32
        ready["prefill_view"]["moe_prefill_group_size"] = self.v6_ready()["prefill_view"]["moe_prefill_group_size"]
        model.generator.decode_views[16].model.layers[0].attention._decode_conv_batch_packed = True
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_conv_pack"
        )
        for native in [
            model.generator.model,
            model.generator.prefill_view.model,
            *[view.model for view in model.generator.decode_views.values()],
        ]:
            for layer in native.layers:
                if layer.kind == "linear_attention":
                    del layer.attention._decode_conv_batch_packed
        model.generator.decode_views[4].model.layers[0].mlp = SimpleNamespace(prefill_group_size=256)
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_prefill_moe_group"
        )
        del model.generator.decode_views[4].model.layers[0].mlp
        del ready["prefill_view"]["moe_prefill_group_size"]
        prefill_layers[3].attention._decode_fold_vectorized = True
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_compact_decode"
        )
        prefill_layers[3].attention._decode_fold_vectorized = False
        owner_layers[3].attention._decode_fold_vectorized = False
        self.assertEqual(
            execution_metadata(model, ready["prefill_view"], True)["execution_version"], "unqualified_compact_decode"
        )

    def test_v5_vector_and_bridge_stored_geometry(self):
        for batch in (4, 8, 16, 32):
            qsa = WORK.vector_qsa_payload(batch)
            self.assertEqual(qsa["qsa_vector_q_permute_boundary"], 48 * (batch * 98304 + 98304))
            self.assertEqual(qsa["qsa_vector_head_pad_boundary"], 48 * (98304 + 524288))
            self.assertEqual(qsa["qsa_vector_untilize_boundary"], 48 * (1048576 + 32768 * batch))
            self.assertEqual(qsa["qsa_vector_output_tilize_boundary"], 48 * (3072 * batch + 98304 * batch))
            bridge = WORK.compact_decode_bridge_payload(batch)
            if batch < 32:
                self.assertEqual(bridge["compact_decode_device_input_slice_boundary"], 4 * 2 * batch * 8257 * 4)
            else:
                self.assertEqual(
                    bridge,
                    {
                        "owner_decode_logits_copy_boundary": 4 * 2 * 32 * 62080 * 2,
                        "decode_owner_token_reshape_boundary": 4 * 2 * 32 * 4,
                    },
                )
        native = WORK.account_events(
            [self.cyclic_ready(1), dict(phase="decode", physical_batch=1, positions=[4096], suffix_lengths=[1])]
        )
        self.assertEqual(native["decode"]["dram_bytes"], 7472666624)

    def cyclic_ready(self, batch):
        partitions = [dict(layer=i, kind="cyclic", work=8192, workers=110) for i in range(3, 48, 4)]
        if batch == 32:
            ready = self.v2_ready()
            ready["execution_version"] = "compact_view_prefix_broadcast_cyclic_v4"
            ready["gdn_recurrence_broadcast"] = [dict(layer=i, enabled=True) for i in range(48) if i % 4 != 3]
            ready["prefill_view"]["gdn_recurrence_broadcast"] = ready["gdn_recurrence_broadcast"]
            ready["prefill_view"]["pooled_gather_partitions"] = copy.deepcopy(partitions)
        else:
            ready = dict(
                event="ready",
                execution_version="native_b1_gdn_cyclic_v3",
                physical_batch=1,
                layers=48,
                layer_indices=list(range(48)),
                context=262144,
                prefill_view=None,
                indexer_prefix_cap=False,
                precision=self.v2_ready()["prefill_view"]["precision"],
                gdn_recurrence_broadcast=[dict(layer=i, enabled=False) for i in range(48) if i % 4 != 3],
            )
        ready["pooled_gather_partitions"] = partitions
        return ready

    def test_cyclic_versions_add_only_known_gather_metadata(self):
        for batch, prior in ((1, "native_b1_gdn_payload_v2"), (32, "compact_view_prefix_broadcast_v3")):
            for active in (1, batch):
                ready = self.cyclic_ready(batch)
                event = dict(
                    phase="decode", physical_batch=batch, positions=[4096] * active, suffix_lengths=[1] * active
                )
                result = WORK.account_events([ready, event])
                historical = WORK.account_events([dict(ready, execution_version=prior), event])
                positions = 4 * 12 * batch * 110 * 128
                tables = 4 * 12 * active * 110 * 544
                self.assertEqual(
                    result["decode"]["dram_bytes"] - historical["decode"]["dram_bytes"], positions + tables
                )
                self.assertEqual(
                    result["decode"]["dram_bytes_routing_lower_estimate"]
                    - historical["decode"]["dram_bytes_routing_lower_estimate"],
                    positions + tables,
                )
                categories = result["details"]["decode_dram_bytes_by_category"]
                self.assertEqual(categories["qsa_pooled_gather_positions_read"], positions)
                self.assertEqual(categories["qsa_pooled_gather_page_table_read"], tables)
                for name, value in historical["details"]["decode_dram_bytes_by_category"].items():
                    self.assertEqual(categories[name], value)

    def test_cyclic_prefill_uses_view_partitions_without_changing_useful_flops(self):
        ready = self.cyclic_ready(32)
        for partition in ready["prefill_view"]["pooled_gather_partitions"]:
            partition["workers"] = 100
        event = dict(
            phase="prefill",
            physical_batch=32,
            positions=[0],
            suffix_lengths=[128],
            physical_chunks=WORK.serial_view_chunks([0], [128]),
        )
        current = WORK.account_events([ready, event])
        prior = WORK.account_events([dict(ready, execution_version="compact_view_prefix_broadcast_v3"), event])
        self.assertEqual(current["prefill"], prior["prefill"])
        self.assertEqual(
            current["details"]["prefill_pooled_gather_metadata_bytes"],
            dict(
                qsa_pooled_gather_positions_read=4 * 12 * 100 * 128, qsa_pooled_gather_page_table_read=4 * 12 * 4 * 32
            ),
        )

    def test_cyclic_partition_missing_partial_or_step_conflict_rejected(self):
        event = dict(phase="decode", physical_batch=32, positions=[4096], suffix_lengths=[1])
        for scope in ("owner", "view"):
            for change in ("missing", "partial", "kind", "workers"):
                ready = self.cyclic_ready(32)
                target = ready if scope == "owner" else ready["prefill_view"]
                if change == "missing":
                    del target["pooled_gather_partitions"]
                elif change == "partial":
                    target["pooled_gather_partitions"].pop()
                else:
                    target["pooled_gather_partitions"][0][change] = "contiguous" if change == "kind" else 0
                with self.subTest(scope=scope, change=change), self.assertRaises(ValueError):
                    WORK.account_events([ready, event])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            WORK.account_events([self.cyclic_ready(32), dict(event, pooled_gather_partitions=[])])

    def test_one_active_prompt_useful_work_is_batch_invariant(self):
        results = [
            WORK.account_events([dict(phase="prefill", physical_batch=batch, positions=[0], suffix_lengths=[8])])
            for batch in (1, 32)
        ]
        self.assertEqual(results[0]["prefill"]["flops"], results[1]["prefill"]["flops"])
        self.assertGreater(
            results[1]["details"]["prefill_executed_matrix_flops_estimate"],
            results[0]["details"]["prefill_executed_matrix_flops_estimate"],
        )
        for result in results:
            by_fidelity = result["details"]["prefill_useful_matrix_flops_by_fidelity"]
            self.assertEqual(result["prefill"]["flops"], sum(by_fidelity.values()))
            ideal = sum(value * WORK.FRACTION[fidelity] / WORK.PEAK_LOFI for fidelity, value in by_fidelity.items())
            self.assertAlmostEqual(result["prefill"]["peak_flops_per_second"], result["prefill"]["flops"] / ideal)
            row = result["details"]["event_work"][0]
            self.assertEqual(
                row["useful_matrix_flops_by_category"]["qsa_sparse_qk_pv"],
                results[0]["details"]["event_work"][0]["useful_matrix_flops_by_category"]["qsa_sparse_qk_pv"],
            )

    def test_4096_chunk_conservation(self):
        for batch, maximum, expected_count in ((1, 512, 8), (32, 8, 512)):
            with self.subTest(batch=batch):
                chunks = WORK.generator_chunks([0] * batch, [4096] * batch, batch)
                self.assertEqual(len(chunks), expected_count)
                self.assertEqual(max(c["chunk_length"] for c in chunks), maximum)
                self.assertEqual(sum(c["chunk_length"] * sum(c["active"]) for c in chunks), batch * 4096)
                result = WORK.account_events(
                    [dict(phase="prefill", physical_batch=batch, positions=[0] * batch, suffix_lengths=[4096] * batch)]
                )
                self.assertEqual(result["details"]["physical_tokens"], batch * 4096)
                self.assertEqual(result["details"]["active_tokens"], batch * 4096)
                self.assertGreater(result["prefill"]["flops"], 0)

    def test_unequal_prompt_lengths_preserve_every_token(self):
        chunks = WORK.generator_chunks([0, 7, 100], [17, 9, 1], 32)
        self.assertEqual(sum(c["chunk_length"] * sum(c["active"]) for c in chunks), 27)
        self.assertTrue(all(len(c["active"]) == 32 for c in chunks))
        self.assertTrue(
            all(p == 0 for c in chunks for p, active in zip(c["start_positions"], c["active"]) if not active)
        )

    def test_missing_authoritative_positions_rejected(self):
        event = dict(phase="decode", physical_batch=1, suffix_lengths=[1])
        with self.assertRaisesRegex(ValueError, "authoritative"):
            WORK.account_events([event])

    def test_missing_positions_rejected_with_optimized_python(self):
        script = """import importlib.util,sys
s=importlib.util.spec_from_file_location("work",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
try: m.account_events([dict(phase="decode",physical_batch=1,suffix_lengths=[1])])
except ValueError: pass
else: raise RuntimeError("missing positions accepted under -O")
"""
        subprocess.run([sys.executable, "-O", "-c", script, str(SOURCE)], check=True)

    def test_submitted_events_do_not_double_count_completed_work(self):
        event = dict(phase="decode", physical_batch=1, positions=[4096], suffix_lengths=[1])
        result = WORK.account_events([dict(event, event="submitted"), dict(event, event="step")])
        self.assertEqual(result["details"]["completed_decode_events"], 1)
        self.assertEqual(
            result["decode"]["dram_bytes"], sum(result["details"]["decode_dram_bytes_by_category"].values())
        )

    def test_legacy_default_preserves_failed_run_payload(self):
        for batch, expected in ((1, 7151203328), (32, 150035398656)):
            event = dict(phase="decode", physical_batch=batch, positions=[4096] * batch, suffix_lengths=[1] * batch)
            default = WORK.account_events([event])
            explicit = WORK.account_events([dict(event="ready", execution_version="legacy"), event])
            self.assertEqual(default, explicit)
            self.assertEqual(default["decode"]["dram_bytes"], expected)

    def test_new_execution_requires_ready_version(self):
        event = dict(
            phase="decode",
            physical_batch=32,
            positions=[4096],
            suffix_lengths=[1],
            execution_version="compact_decode_v1",
        )
        with self.assertRaisesRegex(ValueError, "ready record"):
            WORK.account_events([event])
        with self.assertRaisesRegex(ValueError, "execution versions"):
            WORK.account_events([dict(event="ready", execution_version="unknown"), event])

    def test_compact_b1_and_prefill_useful_work_unchanged(self):
        for phase, batch, length in (("decode", 1, 1), ("prefill", 1, 4096), ("prefill", 32, 8)):
            event = dict(phase=phase, physical_batch=batch, positions=[0] * batch, suffix_lengths=[length] * batch)
            legacy = WORK.account_events([event])
            compact = WORK.account_events([dict(event="ready", execution_version="compact_decode_v1"), event])
            self.assertEqual(legacy["prefill"]["flops"], compact["prefill"]["flops"])
            self.assertEqual(legacy["decode"]["dram_bytes"], compact["decode"]["dram_bytes"])
            self.assertEqual(legacy["details"]["physical_tokens"], compact["details"]["physical_tokens"])

    def test_packed_route_payload_matches_real_tile_reads(self):
        for unique in (10, 165, 320):
            terms = WORK.packed_moe_payload(unique, 320)
            self.assertEqual(terms["packed_active_expert_weights"], unique * 2764800)
            self.assertEqual(terms["packed_expert_output_tile_writes"], unique * 87040)
            self.assertEqual(terms["packed_compact_tile_reads"], 320 * 87040)
            self.assertEqual(sum(terms.values()), unique * 2851840 + 320 * (102400 + 4))
        lower = WORK.packed_moe_payload(1, 32)
        upper = WORK.packed_moe_payload(320, 320)
        self.assertLess(sum(lower.values()), sum(upper.values()))
        with self.assertRaises(ValueError):
            WORK.packed_moe_payload(1, 320)

    def test_compact_bounds_and_metadata_are_explicit(self):
        # One active request still executes all32 physical rows, including scratch.
        event = dict(phase="decode", physical_batch=32, positions=[4096], suffix_lengths=[1])
        result = WORK.account_events([dict(event="ready", execution_version="compact_decode_v1"), event])
        row = result["details"]["event_work"][0]
        bounds = row["decode_routing_bounds"]
        self.assertFalse(bounds["counts_observed"])
        self.assertEqual(bounds["physical_rows"], 32)
        self.assertEqual(bounds["selected_endpoint"], "upper")
        self.assertEqual(bounds["nonzero_routes_per_layer"], {"lower": 32, "upper": 320})
        self.assertEqual(bounds["unique_experts_per_layer"], {"lower": 1, "upper": 320})
        self.assertEqual(result["decode"]["estimate_kind"], "analytical_upper_routing")
        self.assertEqual(result["decode"]["dram_bytes"], sum(row["decode_dram_bytes_by_category"].values()))
        self.assertEqual(
            result["decode"]["dram_bytes"] - result["decode"]["dram_bytes_routing_lower_estimate"],
            bounds["upper_bytes"] - bounds["lower_bytes"],
        )
        traffic = row["decode_dram_bytes_by_category"]
        self.assertNotIn("lane_separate_active_expert_weights", traffic)
        self.assertNotIn("lane_separate_router_shared_weights", traffic)
        self.assertNotIn("expert_dispatch_input_and_output", traffic)
        self.assertEqual(traffic["packed_router_shared_weights"], 1006632960)
        self.assertGreater(traffic["meshops_singleton_layout_conversions"], 0)
        self.assertGreater(traffic["compact_head_input_layout_conversions"], 0)

    def test_compaction_conversion_bytes_include_padded_and_logical_storage(self):
        # BF16 [1,32,1,2560] -> RM view -> TILE [1,1,32,2560].
        expected = 32 * 32 * 2560 * 2 + 3 * 32 * 2560 * 2
        self.assertEqual(WORK.compact_roundtrip_payload(32, 2560, 2), expected)
        self.assertEqual(WORK.compact_roundtrip_payload(32, 2560, 2, expand=True), expected)

    @staticmethod
    def v2_ready():
        return dict(
            event="ready",
            execution_version="compact_view_prefix_v2",
            physical_batch=32,
            layers=48,
            indexer_prefix_cap=True,
            prefill_view=dict(
                execution_batch=1,
                server_slots=32,
                last_token_only=True,
                buckets=WORK.VIEW_BUCKETS,
                context=262144,
                layer_indices=list(range(48)),
                precision=json.loads(
                    (SOURCE.parents[1] / "doc/datatype_sweep/selected_precision_config.json").read_text()
                ),
                layers=[
                    dict(
                        layer=i,
                        ple_cache_shared=True,
                        public_state={
                            name: dict(shared=True, buffer_address=1000 + i)
                            for name in (
                                ("packed_cache", "raw_key_cache", "pooled_cache")
                                if i % 4 == 3
                                else ("recurrent_bank", "conv_bank")
                            )
                        },
                    )
                    for i in range(48)
                ],
            ),
        )

    def test_v2_requires_attested_ready_features(self):
        event = dict(phase="decode", physical_batch=32, positions=[4096], suffix_lengths=[1])
        ready = self.v2_ready()
        for missing in ("prefill_view", "indexer_prefix_cap"):
            bad = dict(ready)
            del bad[missing]
            with self.assertRaises(ValueError):
                WORK.account_events([bad, event])

    def test_v2_serial_prefill_has_no_inactive_scratch(self):
        event = dict(
            phase="prefill", physical_batch=32, server_slots=32, positions=[0, 17, 100], suffix_lengths=[4096, 131, 0]
        )
        with self.assertRaisesRegex(ValueError, "observed physical_chunks"):
            WORK.account_events([self.v2_ready(), event])
        chunks = WORK.serial_view_chunks(event["positions"], event["suffix_lengths"])
        event["physical_chunks"] = chunks
        result = WORK.account_events([self.v2_ready(), event])
        row = result["details"]["event_work"][0]
        self.assertEqual(row["active_tokens"], 4227)
        self.assertEqual(row["physical_tokens"], 4227)
        self.assertEqual(row["batch"], 32)
        self.assertEqual(row["server_slots"], 32)
        self.assertEqual(row["execution_batches"], [1])
        self.assertEqual(len(chunks), 35)  # 32x128; then128,2,1
        self.assertEqual([c["chunk_length"] for c in chunks[-3:]], [128, 2, 1])
        explicit = WORK.account_events([self.v2_ready(), dict(event, physical_chunks=chunks)])
        self.assertEqual(result, explicit)
        self.assertEqual(row["matrix_flops_by_category"]["lm_head"], 35 * 2 * 2560 * 248320)
        broken = [dict(c) for c in chunks]
        broken[-1]["start_positions"] = [0]
        with self.assertRaisesRegex(ValueError, "authoritative per-row schedule"):
            WORK.account_events([self.v2_ready(), dict(event, physical_chunks=broken)])

    def test_v2_rejects_reduced_layers_precision_and_unshared_state(self):
        ready = self.v2_ready()
        bad = copy.deepcopy(ready)
        bad["layers"] = 4
        with self.assertRaisesRegex(ValueError, "all48"):
            WORK.validate_v2_ready(bad)
        bad = copy.deepcopy(ready)
        bad["prefill_view"]["layer_indices"][-1] = 46
        with self.assertRaisesRegex(ValueError, "all48"):
            WORK.validate_v2_ready(bad)
        bad = copy.deepcopy(ready)
        bad["prefill_view"]["precision"]["weight_groups"]["lm_head"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "selected policy"):
            WORK.validate_v2_ready(bad)
        bad = copy.deepcopy(ready)
        bad["prefill_view"]["layers"][3]["public_state"]["pooled_cache"]["shared"] = False
        with self.assertRaisesRegex(ValueError, "shared with the owner"):
            WORK.validate_v2_ready(bad)
        bad = copy.deepcopy(ready)
        bad["prefill_view"]["layers"][1]["ple_cache_shared"] = False
        with self.assertRaisesRegex(ValueError, "shared with the owner"):
            WORK.validate_v2_ready(bad)

    def test_v2_indexer_cap_boundary_formula(self):
        for position, sequence, expected in (
            (0, 1, 512),
            (2047, 1, 512),
            (2048, 4, 544),
            (4095, 1, 1024),
            (4096, 1, 1024),
            (262143, 1, 65536),
        ):
            self.assertEqual(WORK.indexer_prefix_columns(position, sequence), expected)
        with self.assertRaises(ValueError):
            WORK.indexer_prefix_columns(262143, 2)

    def test_v2_256_geometry_is_selected_from_actual_ready(self):
        results = {}
        for maximum in (128, 256):
            ready = self.v2_ready()
            buckets = WORK.VIEW_BUCKETS + ([256] if maximum == 256 else [])
            ready["prefill_view"]["buckets"] = buckets
            chunks = WORK.serial_view_chunks([0], [4096], buckets=buckets)
            event = dict(
                phase="prefill", physical_batch=32, positions=[0], suffix_lengths=[4096], physical_chunks=chunks
            )
            results[maximum] = WORK.account_events([ready, event])
            result = results[maximum]
            self.assertEqual(result["details"]["physical_tokens"], 4096)
            self.assertEqual(len(chunks), 4096 // maximum)
            self.assertEqual(result["prefill"]["execution_geometry"], f"serial_real_rows_batch1_max{maximum}")
            self.assertEqual(result["details"]["prefill_view_buckets"], buckets)
        self.assertEqual(
            results[128]["details"]["prefill_matrix_flops_by_category"]["lm_head"],
            2 * results[256]["details"]["prefill_matrix_flops_by_category"]["lm_head"],
        )
        with self.assertRaisesRegex(ValueError, "authoritative per-row schedule"):
            WORK.account_events([self.v2_ready(), event])

    def test_v2_reduces_native_indexer_work_not_full_mask_scans(self):
        event = dict(phase="decode", physical_batch=32, positions=[4096] * 32, suffix_lengths=[1] * 32)
        v1 = WORK.account_events([dict(event="ready", execution_version="compact_decode_v1"), event])
        v2 = WORK.account_events([self.v2_ready(), event])
        a = v1["details"]["event_work"][0]
        b = v2["details"]["event_work"][0]
        self.assertEqual(b["indexer_columns"][0]["columns_per_row"], [1024] * 32)
        self.assertEqual(
            a["matrix_flops_by_category"]["qsa_dense_indexer"], b["matrix_flops_by_category"]["qsa_dense_indexer"] * 64
        )
        ta, tb = a["decode_dram_bytes_by_category"], b["decode_dram_bytes_by_category"]
        self.assertEqual(ta["qsa_dense_pooled_key_reads"], tb["qsa_dense_pooled_key_reads"] * 64)
        self.assertEqual(tb["qsa_indexer_score_write_read"], 4 * 12 * 32 * (32 * 1024 * 2 + 3 * 65536 * 2))
        self.assertEqual(a["decode_routing_bounds"], b["decode_routing_bounds"])
        self.assertEqual(a["useful_matrix_flops"], b["useful_matrix_flops"])

    def test_v3_attests_all_broadcast_layers_and_refines_dram_ledger(self):
        ready = self.v2_ready()
        ready["prefill_view"]["buckets"] = WORK.VIEW_BUCKETS + [256]
        events = [
            dict(
                phase="prefill",
                physical_batch=32,
                positions=[0],
                suffix_lengths=[4096],
                physical_chunks=WORK.serial_view_chunks([0], [4096], buckets=ready["prefill_view"]["buckets"]),
            ),
            dict(phase="decode", physical_batch=32, positions=[4096], suffix_lengths=[1]),
        ]
        v2 = WORK.account_events([ready, *events])
        flags = [dict(layer=i, enabled=True) for i in range(48) if i % 4 != 3]
        ready["execution_version"] = "compact_view_prefix_broadcast_v3"
        ready["gdn_recurrence_broadcast"] = flags
        ready["prefill_view"]["gdn_recurrence_broadcast"] = copy.deepcopy(flags)
        v3 = WORK.account_events([ready, *events])
        self.assertEqual(v3["prefill"], v2["prefill"])
        delta = sum(WORK.gdn_recurrence_payload(32).values()) - 4 * 32 * 36 * 4 * WORK.matrix(12, 128, "float32")
        self.assertEqual(v3["decode"]["dram_bytes"] - v2["decode"]["dram_bytes"], delta)
        self.assertEqual(
            v3["decode"]["dram_bytes_routing_lower_estimate"] - v2["decode"]["dram_bytes_routing_lower_estimate"],
            delta,
        )
        a, b = v3["details"]["event_work"][1], v2["details"]["event_work"][1]
        self.assertEqual(a["useful_matrix_flops"], b["useful_matrix_flops"])
        self.assertEqual(a["decode_routing_bounds"], b["decode_routing_bounds"])
        ta, tb = a["decode_dram_bytes_by_category"], b["decode_dram_bytes_by_category"]
        self.assertNotIn("gdn_projection_and_recurrence_intermediates", ta)
        self.assertEqual(
            ta["gdn_packed_projection_output_and_boundary"],
            4 * 36 * (WORK.matrix(32, 4160, "bfloat16") + 32 * WORK.matrix(32, 4160, "bfloat16")),
        )
        self.assertEqual(ta["gdn_state_bank_work_recurrence"], tb["gdn_state_bank_work_recurrence"])
        self.assertEqual(ta["meshops_singleton_layout_conversions"], tb["meshops_singleton_layout_conversions"])
        self.assertEqual(v3["details"]["execution_version"], "compact_view_prefix_broadcast_v3")
        for target in (ready, ready["prefill_view"]):
            saved = target["gdn_recurrence_broadcast"]
            target["gdn_recurrence_broadcast"] = saved[:-1]
            with self.assertRaisesRegex(ValueError, "all36"):
                WORK.account_events([ready, *events])
            target["gdn_recurrence_broadcast"] = saved
        ready["execution_version"] = "compact_view_prefix_v2"
        with self.assertRaisesRegex(ValueError, "historical v2"):
            WORK.account_events([ready, *events])

    def test_v3_native_gdn_terms_count_each_boundary_once(self):
        terms = WORK.gdn_recurrence_payload(32)
        factor = 4 * 36 * 32
        expected = {
            "gdn_recurrence_final_operand_writes": 57344,
            "gdn_recurrence_native_operand_reads": 2162688,
            "gdn_recurrence_native_output_writes": 196608,
            "gdn_recurrence_output_typecast_read_write": 294912,
            "gdn_recurrence_output_padding_read_write": 196608,
            "gdn_recurrence_gated_norm_input_reads": 98304,
        }
        self.assertEqual(terms, {name: amount * factor for name, amount in expected.items()})
        self.assertEqual(terms["gdn_recurrence_native_operand_reads"], 9965666304)
        self.assertEqual(sum(terms.values()), 3006464 * factor)
        with self.assertRaisesRegex(ValueError, "Unsupported GDN"):
            WORK.gdn_recurrence_payload(2)

    def test_current_native_b1_preserves_execution_with_explicit_dram_placement(self):
        ready = dict(
            event="ready",
            execution_version="native_b1_gdn_payload_v2",
            physical_batch=1,
            layers=48,
            layer_indices=list(range(48)),
            context=262144,
            prefill_view=None,
            indexer_prefix_cap=False,
            precision=self.v2_ready()["prefill_view"]["precision"],
            gdn_recurrence_broadcast=[dict(layer=i, enabled=False) for i in range(48) if i % 4 != 3],
        )
        events = [
            dict(phase="prefill", physical_batch=1, positions=[0], suffix_lengths=[4096]),
            dict(phase="decode", physical_batch=1, positions=[4096], suffix_lengths=[1]),
        ]
        legacy = WORK.account_events(events)
        current = WORK.account_events([ready, *events])
        self.assertEqual(current["prefill"], legacy["prefill"])
        a = current["details"]["event_work"][1]["decode_dram_bytes_by_category"]
        b = legacy["details"]["event_work"][1]["decode_dram_bytes_by_category"]
        self.assertEqual(a["gdn_packed_projection_output_and_boundary"], 0)
        self.assertEqual(a["gdn_recurrence_final_operand_writes"], 4 * 36 * 13 * 4096)
        self.assertEqual(a["gdn_recurrence_native_operand_reads"], 4 * 36 * 48 * 10 * 4096)
        self.assertEqual(a["gdn_state_bank_work_recurrence"], b["gdn_state_bank_work_recurrence"])
        self.assertEqual(
            current["decode"]["dram_bytes"] - legacy["decode"]["dram_bytes"],
            sum(WORK.gdn_recurrence_payload(1).values()) - b["gdn_projection_and_recurrence_intermediates"],
        )
        for key in ("precision", "layer_indices", "gdn_recurrence_broadcast"):
            bad = dict(ready)
            del bad[key]
            with self.assertRaises(ValueError):
                WORK.account_events([bad, *events])
        bad = copy.deepcopy(ready)
        bad["gdn_recurrence_broadcast"][0]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "all36"):
            WORK.account_events([bad, *events])


if __name__ == "__main__":
    unittest.main(verbosity=2)
