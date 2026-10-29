# SPDX-License-Identifier: Apache-2.0
"""Pure source-method fixtures, NOT a vLLM schema/admission or hardware gate.

Executes verified method ASTs with explicit logical block/spec fixtures. It
imports no vLLM, TTNN, Torch or inference package and edits no installed source.
"""

import argparse
import ast
import hashlib
import json
import math
import zipfile
from pathlib import Path
from types import MethodType, SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-archive", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
EXPECTED_ARCHIVE_SHA256 = "1c327709dc8b1bce978ee9a22e47afb0923d017d1a1e1afea933e70a85ad3504"
assert hashlib.sha256(args.source_archive.read_bytes()).hexdigest() == EXPECTED_ARCHIVE_SHA256
with zipfile.ZipFile(args.source_archive) as archive:
    MANIFEST = json.loads(archive.read("manifest.json"))
    assert MANIFEST["image"].endswith("@sha256:41aa0687bc668186baf9c4ed7942e135b60cf8828d9bbd67418cf53993a3d612")
    assert MANIFEST["vllm_version"] == "0.30.0+empty"
    assert set(archive.namelist()) == {"manifest.json"} | {r["path"] for r in MANIFEST["files"]}
    SOURCES = {}
    for row in MANIFEST["files"]:
        data = archive.read(row["path"])
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        SOURCES[row["path"]] = data.decode()


def extract(path, name, cls=None):
    tree = ast.parse(SOURCES[path])
    scope = (
        tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    )
    fn = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), GLOBALS)
    return GLOBALS[name]


class LogicalSpec(SimpleNamespace):
    """Only algorithm fixture fields; does not stand in for real MambaSpec."""


class Block:
    def __init__(self, ident):
        self.block_id = ident
        self.is_null = ident == 0
        self.block_hash = None


class Pool:
    hash_block_size = 512
    num_gpu_blocks = 1470

    def __init__(self):
        self.allocated = []

    def get_new_blocks(self, n):
        result = [Block(100 + i) for i in range(n)]
        self.allocated.extend(result)
        return result


GLOBALS = dict(MambaSpec=LogicalSpec, cdiv=lambda a, b: (a + b - 1) // b)
pos = extract("vllm/v1/kv_cache_interface.py", "get_mamba_prefill_checkpoint_position")
valid = extract("vllm/v1/kv_cache_interface.py", "is_mamba_prefill_checkpoint_valid")
MANAGER = "vllm/v1/core/single_type_kv_cache_manager.py"
needs = extract(MANAGER, "_needs_internal_checkpoint", "MambaManager")
predict = extract(MANAGER, "get_num_blocks_to_allocate", "MambaManager")
allocate = extract(MANAGER, "allocate_new_blocks", "MambaManager")


def fixture(checkpoints, start):
    null = Block(0)
    blocks = [null] * (start // 512)
    if blocks:
        blocks[-1] = Block(1)
    if checkpoints and len(blocks) >= 2:
        blocks[-2] = Block(2)
    obj = SimpleNamespace(
        kv_cache_spec=LogicalSpec(block_size=512, prefill_checkpoint_alignment=512),
        mamba_cache_mode="align",
        block_size=512,
        num_speculative_blocks=0,
        has_prefill_checkpoint_blocks=bool(checkpoints),
        req_to_blocks={"r": blocks},
        cached_blocks_this_step=set(),
        _partial_hit_reqs={},
        _checkpoints={},
        _allocated_block_reqs={"r"} if start else set(),
        _null_block=null,
        last_state_block_idx={},
        block_pool=Pool(),
        drop_eagle_checkpoint_block=False,
        _has_partial_local_hit=lambda *args: False,
        _get_num_evictable_blocks=lambda *args: 0,
    )
    obj._needs_internal_checkpoint = MethodType(needs, obj)
    return obj


def run(checkpoints, start, end):
    obj = fixture(checkpoints, start)
    estimate = predict(obj, "r", end, [], start, 0, end)
    returned = allocate(obj, "r", end, end)
    actual = len(obj.block_pool.allocated)
    return dict(
        checkpoints=checkpoints,
        start=start,
        end=end,
        predicted_physical_blocks=estimate,
        actual_physical_blocks=actual,
        returned_logical_entries=len(returned),
        returned_null_entries=sum(b.is_null for b in returned),
        returned_block_ids=[b.block_id for b in returned],
        actual_new_block_ids=[b.block_id for b in obj.block_pool.allocated],
        checkpoint=obj._checkpoints.get("r"),
        tail_ids=[b.block_id for b in obj.req_to_blocks["r"][-2:]],
    )


rows = [run(c, s, e) for c in [0, 1] for s, e in [(0, 4096), (4096, 8192), (12288, 15872), (15872, 16896)]]
assert (
    next(r for r in rows if (r["checkpoints"], r["start"], r["end"]) == (0, 4096, 8192))["predicted_physical_blocks"]
    == 1
)
bug = next(r for r in rows if (r["checkpoints"], r["start"], r["end"]) == (1, 4096, 8192))
assert (bug["predicted_physical_blocks"], bug["actual_physical_blocks"], bug["returned_null_entries"]) == (8, 2, 6)
assert bug["returned_logical_entries"] == 8
assert bug["returned_block_ids"] == [0] * 6 + bug["actual_new_block_ids"]
assert pos(16384, 512, False) == 15872 and pos(32768, 512, False) == 32256
assert valid(12288, 16384, 15872, 512, 512, 512)
for n in range(513, 18000):
    cp = pos(n, 512, False)
    assert (n + 511) // 512 - 2 == (cp - 1) // 512

# Execute the installed worker's sizing body with explicit host-only fixtures.
# The upstream admission floor is stubbed out to isolate its reservation terms;
# this is not a genuine framework admission test.
worker = extract("vllm_tt_plugin/worker.py", "get_num_available_blocks_tt")


def worker_blocks(checkpoints, hook):
    class ModelFixture:
        _HYBRID_KV_CACHE_GROUPS_ENABLED = True

        @staticmethod
        def get_max_tokens_all_users(**kwargs):
            return hook

    groups = [
        SimpleNamespace(
            kv_cache_spec=LogicalSpec(
                num_speculative_blocks=0, num_prefill_checkpoint_blocks=checkpoints, mamba_cache_mode="align"
            )
        )
        for _ in range(3)
    ]
    GLOBALS.update(
        math=math,
        logger=SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        get_tt_data_parallel_size=lambda cfg: 1,
        get_model_architecture=lambda cfg: (ModelFixture, None),
        get_tt_per_lane_max_num_seqs=lambda cfg: 32,
        get_tt_output_tokens_per_step=lambda cfg: 1,
        kv_cache_spec_from_model_class=lambda *a: {"fixture": True},
        _kv_cache_groups_for_sizing=lambda *a: (None, groups),
        _upstream_admission_blocks=lambda *a: 0,
        is_tt_block_output_model=lambda *a: False,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        model_config=SimpleNamespace(
            model="Qwen/Qwen3.8-27B",
            max_model_len=262144,
            original_max_model_len=262144,
            get_sliding_window=lambda: None,
        ),
    )
    return worker(config, num_devices=4)


assert worker_blocks(0, 637440) == 1470
assert worker_blocks(1, 588288) == 1470
assert worker_blocks(1, 637440) == 1566
assert worker_blocks(0, 590848) == 1379

report = dict(
    scope=__doc__,
    actual_image=MANIFEST["image"],
    source_hashes_verified=True,
    cases=rows,
    source_predictor_mismatch_reproduced=True,
    checkpoints_are_penultimate_not_all512=True,
    sizing=dict(
        control_max_ids=1379,
        shared_physical_ids=1470,
        old_reservation=225,
        new_reservation=321,
        shared_default_hook_units=1245,
        shared_checkpoint1_hook_units=1149,
        shared_checkpoint1_old_policy_equivalent_IDs=1374,
        control_default_hook_units=1154,
    ),
    limits=[
        "Logical method fixtures are not schema admission, actual package execution, end-to-end caching or device validation.",
        "No partial-hit/CoW/speculative cases are claimed; these require separate coverage before enabling.",
        "Worker sizing fixture excludes the upstream admission floor; genuine framework admission remains untested.",
    ],
)
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
