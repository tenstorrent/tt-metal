-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
--
-- SPDX-License-Identifier: Apache-2.0

-- Additive migration: Add trace_run support to existing ttnn_ops_v2_5 schema
--
-- This migration is:
--   - Non-destructive: No DROP, no ALTER COLUMN TYPE, no data deletion
--   - Idempotent: Safe to run multiple times (IF NOT EXISTS everywhere)
--   - Backward-compatible: Existing tables and queries continue to work
--
-- What it adds:
--   1. trace_run          — First-class entity for each tracer execution
--   2. trace_run_config   — Junction: which configs came from which trace (replaces ttnn_configuration_model)
--   3. model_name column  — Human-readable name on ttnn_model
--
-- What it does NOT do:
--   - Drop ttnn_configuration_model (kept for backward compat; migrate data below)
--   - Drop ttnn_hardware (still referenced by ttnn_configuration.hardware_id)
--   - Change config_hash computation (same identity contract)
--   - Create a new schema (everything stays in ttnn_ops_v2_5)

-- 1. Add model_name to ttnn_model (display name, not part of unique constraint)
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'ttnn_ops_v2_5'
          AND table_name = 'ttnn_model'
          AND column_name = 'model_name'
    ) THEN
        ALTER TABLE ttnn_ops_v2_5.ttnn_model
            ADD COLUMN model_name TEXT;

        -- Backfill: derive model_name from source_file or hf_model_identifier
        UPDATE ttnn_ops_v2_5.ttnn_model
        SET model_name = COALESCE(
            -- Prefer HF model name (last segment after /)
            CASE WHEN hf_model_identifier IS NOT NULL
                 THEN SPLIT_PART(hf_model_identifier, '/', -1)
                 ELSE NULL END,
            -- Fallback: extract from source_file path
            REGEXP_REPLACE(
                REGEXP_REPLACE(source_file, '^.*/([^/]+)/?$', '\1'),
                '\.(py|json)$', ''
            )
        )
        WHERE model_name IS NULL;
    END IF;
END $$;


-- 2. trace_run: one row per tracer execution
CREATE TABLE IF NOT EXISTS ttnn_ops_v2_5.trace_run (
    trace_run_id    SERIAL PRIMARY KEY,
    model_id        INTEGER NOT NULL
                        REFERENCES ttnn_ops_v2_5.ttnn_model(ttnn_model_id),
    board_type      TEXT NOT NULL,
    device_series   TEXT NOT NULL,
    card_count      INTEGER NOT NULL,
    tt_metal_sha    TEXT,
    traced_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config_count    INTEGER,
    notes           TEXT,
    create_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes (idempotent via IF NOT EXISTS)
CREATE INDEX IF NOT EXISTS trace_run_model_idx
    ON ttnn_ops_v2_5.trace_run(model_id);
CREATE INDEX IF NOT EXISTS trace_run_traced_at_idx
    ON ttnn_ops_v2_5.trace_run(traced_at DESC);
CREATE INDEX IF NOT EXISTS trace_run_tt_metal_sha_idx
    ON ttnn_ops_v2_5.trace_run(tt_metal_sha);


-- 3. trace_run_config: junction table (replaces ttnn_configuration_model semantics)
CREATE TABLE IF NOT EXISTS ttnn_ops_v2_5.trace_run_config (
    trace_run_id        INTEGER NOT NULL
                            REFERENCES ttnn_ops_v2_5.trace_run(trace_run_id),
    configuration_id    INTEGER NOT NULL
                            REFERENCES ttnn_ops_v2_5.ttnn_configuration(ttnn_configuration_id),
    execution_count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (trace_run_id, configuration_id)
);

CREATE INDEX IF NOT EXISTS trace_run_config_config_idx
    ON ttnn_ops_v2_5.trace_run_config(configuration_id);


-- 4. Migrate existing ttnn_configuration_model data into trace_run + trace_run_config
--
-- Strategy: For each unique (model_id, hardware on config) pair in the existing
-- junction table, create a synthetic trace_run with tt_metal_sha = NULL and
-- traced_at = first_seen_ts.  Then copy the links.
--
-- This is a one-time migration.  Guard with a check so re-running is safe.
DO $$ BEGIN
    -- Only migrate if trace_run is empty and configuration_model has data
    IF NOT EXISTS (SELECT 1 FROM ttnn_ops_v2_5.trace_run LIMIT 1)
       AND EXISTS (SELECT 1 FROM ttnn_ops_v2_5.ttnn_configuration_model LIMIT 1)
    THEN
        -- Create exactly one trace_run per (model, hardware) group
        INSERT INTO ttnn_ops_v2_5.trace_run
            (model_id, board_type, device_series, card_count, tt_metal_sha, traced_at, notes)
        SELECT
            cm.model_id,
            COALESCE(h.board_type, 'unknown'),
            COALESCE(h.device_series, 'unknown'),
            COALESCE(h.card_count, 1),
            NULL,  -- tt_metal_sha unknown for historical data
            MIN(cm.first_seen_ts),
            'Migrated from ttnn_configuration_model'
        FROM ttnn_ops_v2_5.ttnn_configuration_model cm
        JOIN ttnn_ops_v2_5.ttnn_configuration c
            ON c.ttnn_configuration_id = cm.configuration_id
        LEFT JOIN ttnn_ops_v2_5.ttnn_hardware h
            ON h.ttnn_hardware_id = c.hardware_id
        GROUP BY cm.model_id, h.board_type, h.device_series, h.card_count;

        -- Link configs to their trace_runs
        INSERT INTO ttnn_ops_v2_5.trace_run_config
            (trace_run_id, configuration_id, execution_count)
        SELECT
            tr.trace_run_id,
            cm.configuration_id,
            cm.execution_count
        FROM ttnn_ops_v2_5.ttnn_configuration_model cm
        JOIN ttnn_ops_v2_5.ttnn_configuration c
            ON c.ttnn_configuration_id = cm.configuration_id
        LEFT JOIN ttnn_ops_v2_5.ttnn_hardware h
            ON h.ttnn_hardware_id = c.hardware_id
        JOIN ttnn_ops_v2_5.trace_run tr
            ON tr.model_id = cm.model_id
           AND tr.board_type = COALESCE(h.board_type, 'unknown')
           AND tr.device_series = COALESCE(h.device_series, 'unknown')
           AND tr.card_count = COALESCE(h.card_count, 1)
           AND tr.notes = 'Migrated from ttnn_configuration_model'
        ON CONFLICT DO NOTHING;

        -- Update config_count on each trace_run
        UPDATE ttnn_ops_v2_5.trace_run tr
        SET config_count = (
            SELECT COUNT(*) FROM ttnn_ops_v2_5.trace_run_config trc
            WHERE trc.trace_run_id = tr.trace_run_id
        );

        RAISE NOTICE 'Migrated existing configuration_model data into trace_run tables';
    END IF;
END $$;


-- 5. Verify
SELECT 'trace_run' AS table_name, COUNT(*) AS row_count
    FROM ttnn_ops_v2_5.trace_run
UNION ALL
SELECT 'trace_run_config', COUNT(*)
    FROM ttnn_ops_v2_5.trace_run_config
UNION ALL
SELECT 'ttnn_configuration_model (legacy)', COUNT(*)
    FROM ttnn_ops_v2_5.ttnn_configuration_model;
