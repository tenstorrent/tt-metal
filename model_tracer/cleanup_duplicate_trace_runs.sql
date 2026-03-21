-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
--
-- SPDX-License-Identifier: Apache-2.0

-- Cleanup: collapse duplicate migrated trace_runs into one per (model, hardware)
--
-- The initial migration incorrectly created one trace_run per unique
-- (model, hardware, first_seen_ts) instead of one per (model, hardware).
-- This script consolidates them: picks the lowest trace_run_id as canonical,
-- re-links all trace_run_config rows to it, then deletes the duplicates.
--
-- Only touches rows with notes = 'Migrated from ttnn_configuration_model'.
-- Safe to run multiple times (idempotent).

BEGIN;

-- 1. Find the canonical trace_run_id per (model_id, device_series) group
--    (lowest ID = earliest migrated trace)
CREATE TEMP TABLE canonical_trace_runs AS
SELECT
    model_id,
    board_type,
    device_series,
    card_count,
    MIN(trace_run_id) AS canonical_id
FROM ttnn_ops_v2_5.trace_run
WHERE notes = 'Migrated from ttnn_configuration_model'
GROUP BY model_id, board_type, device_series, card_count;

-- 2. Re-link trace_run_config rows from duplicate traces to the canonical one
INSERT INTO ttnn_ops_v2_5.trace_run_config (trace_run_id, configuration_id, execution_count)
SELECT
    c.canonical_id,
    trc.configuration_id,
    trc.execution_count
FROM ttnn_ops_v2_5.trace_run_config trc
JOIN ttnn_ops_v2_5.trace_run tr ON tr.trace_run_id = trc.trace_run_id
JOIN canonical_trace_runs c
    ON  c.model_id      = tr.model_id
    AND c.board_type    = tr.board_type
    AND c.device_series = tr.device_series
    AND c.card_count    = tr.card_count
WHERE tr.trace_run_id != c.canonical_id
  AND tr.notes = 'Migrated from ttnn_configuration_model'
ON CONFLICT (trace_run_id, configuration_id) DO NOTHING;

-- 3. Delete duplicate (non-canonical) trace_run_config rows
DELETE FROM ttnn_ops_v2_5.trace_run_config
WHERE trace_run_id IN (
    SELECT tr.trace_run_id
    FROM ttnn_ops_v2_5.trace_run tr
    JOIN canonical_trace_runs c
        ON  c.model_id      = tr.model_id
        AND c.board_type    = tr.board_type
        AND c.device_series = tr.device_series
        AND c.card_count    = tr.card_count
    WHERE tr.trace_run_id != c.canonical_id
      AND tr.notes = 'Migrated from ttnn_configuration_model'
);

-- 4. Delete the duplicate trace_runs themselves
DELETE FROM ttnn_ops_v2_5.trace_run
WHERE notes = 'Migrated from ttnn_configuration_model'
  AND trace_run_id NOT IN (SELECT canonical_id FROM canonical_trace_runs);

-- 5. Update config_count on canonical trace_runs
UPDATE ttnn_ops_v2_5.trace_run tr
SET config_count = (
    SELECT COUNT(*)
    FROM ttnn_ops_v2_5.trace_run_config trc
    WHERE trc.trace_run_id = tr.trace_run_id
)
WHERE tr.trace_run_id IN (SELECT canonical_id FROM canonical_trace_runs);

-- 6. Verify
SELECT
    m.source_file,
    m.hf_model_identifier,
    tr.device_series,
    tr.card_count,
    tr.trace_run_id,
    tr.config_count,
    tr.traced_at
FROM ttnn_ops_v2_5.trace_run tr
JOIN ttnn_ops_v2_5.ttnn_model m ON m.ttnn_model_id = tr.model_id
WHERE tr.notes = 'Migrated from ttnn_configuration_model'
ORDER BY m.source_file, tr.device_series;

COMMIT;
