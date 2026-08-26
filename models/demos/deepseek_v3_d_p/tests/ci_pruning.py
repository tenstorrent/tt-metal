# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Collection-time pruning rules for the Blaze prefill suites.

Production prefill is chunked + non_balanced. Cases that exercise a configuration
production never runs are *uncollected* here rather than deselected with a `-k`
expression in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`, so the matrix
keeps its baseline filters and no dead parametrization is left behind for a reader
to reverse-engineer.

Every rule prunes in CI only - a local run still collects everything, which is the
point: these configurations remain useful for local debugging, they are just not
worth a Galaxy job. `conftest.py` owns the `uncollect_if` marker that applies them.

Rules are attached per `pytest.param` row (pytest carries `marks=` through to the
collected item), so the reason a row is not run in CI sits on the row itself.
"""


def _ci_only(pred):
    """Wrap `pred` so it prunes in CI and is a no-op everywhere else."""

    def uncollect_if(**params):
        if not (params["is_ci_env"] or params["is_ci_v2_env"]):
            return False
        return pred(**params)

    return uncollect_if


# GLM-5.1 is fully retired from CI; only GLM-5.2 is a production target. The gate
# cases are kept in the source so the 5.1 config stays locally runnable.
retired_model = _ci_only(lambda **params: True)

# Op-level MoE perf rows. Throughput for these shapes is gated at the model level
# (the chunked-transformer perf legs), so the op-level perf twin buys no signal for
# a Galaxy slot. The matching `pcc-*` row on the same shape is still collected.
perf_row_covered_at_model_level = _ci_only(lambda **params: True)
