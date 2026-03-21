-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
--
-- SPDX-License-Identifier: Apache-2.0

-- Consolidate all n300 trace_runs into a single trace.
--
-- Before: 19 separate trace_runs (one per model+hardware from v2.5 migration)
-- After:  1 n300 trace, 1 p150b trace, 1 tt-galaxy-wh trace (3 total)
--
-- trace_run.model_id is NOT NULL, so we create a combined model entry
-- to represent "all n300 models from the v2.5 baseline load".

BEGIN;

-- 1. Create a combined model entry for the n300 aggregate
INSERT INTO ttnn_ops_v2_5.ttnn_model (source_file, hf_model_identifier, model_family, model_name)
VALUES ('multiple', NULL, 'multi', 'n300_all_models')
RETURNING ttnn_model_id;

-- 2. Create the canonical n300 trace_run (using the new model entry)
INSERT INTO ttnn_ops_v2_5.trace_run
    (model_id, board_type, device_series, card_count, tt_metal_sha, traced_at, notes)
SELECT
    ttnn_model_id,
    'Wormhole',
    'n300',
    1,
    NULL,
    NOW(),
    'Consolidated n300 trace — all models (v2.5 migration)'
FROM ttnn_ops_v2_5.ttnn_model
WHERE source_file = 'multiple' AND model_name = 'n300_all_models'
RETURNING trace_run_id;

-- 3. Copy all unique config links from n300 traces to the canonical trace
INSERT INTO ttnn_ops_v2_5.trace_run_config (trace_run_id, configuration_id, execution_count)
SELECT
    (SELECT trace_run_id FROM ttnn_ops_v2_5.trace_run
     WHERE device_series = 'n300' AND notes = 'Consolidated n300 trace — all models (v2.5 migration)'),
    trc.configuration_id,
    MAX(trc.execution_count)
FROM ttnn_ops_v2_5.trace_run_config trc
JOIN ttnn_ops_v2_5.trace_run tr ON tr.trace_run_id = trc.trace_run_id
WHERE tr.device_series = 'n300'
  AND tr.notes = 'Migrated from ttnn_configuration_model'
GROUP BY trc.configuration_id
ON CONFLICT (trace_run_id, configuration_id) DO NOTHING;

-- 4. Remove config links from the old n300 traces
DELETE FROM ttnn_ops_v2_5.trace_run_config
WHERE trace_run_id IN (
    SELECT trace_run_id FROM ttnn_ops_v2_5.trace_run
    WHERE device_series = 'n300'
      AND notes = 'Migrated from ttnn_configuration_model'
);

-- 5. Delete the old n300 trace_runs
DELETE FROM ttnn_ops_v2_5.trace_run
WHERE device_series = 'n300'
  AND notes = 'Migrated from ttnn_configuration_model';

-- 6. Update config_count on the canonical trace
UPDATE ttnn_ops_v2_5.trace_run
SET config_count = (
    SELECT COUNT(*) FROM ttnn_ops_v2_5.trace_run_config trc
    WHERE trc.trace_run_id = trace_run.trace_run_id
)
WHERE notes = 'Consolidated n300 trace — all models (v2.5 migration)';

-- 7. Verify: should show exactly 3 traces
SELECT
    tr.trace_run_id,
    COALESCE(m.model_name, m.source_file) AS model,
    tr.device_series,
    tr.card_count,
    tr.config_count,
    tr.notes
FROM ttnn_ops_v2_5.trace_run tr
JOIN ttnn_ops_v2_5.ttnn_model m ON m.ttnn_model_id = tr.model_id
ORDER BY tr.trace_run_id;

COMMIT;
