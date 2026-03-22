-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
--
-- SPDX-License-Identifier: Apache-2.0

-- TTNN Ops Database Schema v5
--
-- Key changes from v2_5 / v4:
--   1. trace_run has NO model_id — models are always enumerated per-config
--      via ttnn_configuration_model. No synthetic "multiple" model entries.
--   2. trace_run links to ttnn_hardware by FK (normalized, not inline columns).
--   3. trace_run_model is a derived junction table: which models appeared in
--      a trace (materialized from trace_run_config → ttnn_configuration_model).
--   4. full_config_json on ttnn_configuration is the source of truth; structured
--      columns are best-effort projections for fast filtering.
--
-- Drop and recreate schema
DROP SCHEMA IF EXISTS ttnn_ops_v5 CASCADE;
CREATE SCHEMA ttnn_ops_v5;

-- ---------------------------------------------------------------------------
-- 1. ttnn_operation
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_operation (
    ttnn_operation_id   SERIAL PRIMARY KEY,
    operation_name      TEXT NOT NULL UNIQUE,
    base_operation_name TEXT GENERATED ALWAYS AS (
        REGEXP_REPLACE(operation_name,
            '^(ttnn::experimental::|ttnn::transformer::|experimental::|ttnn::)', '')
    ) STORED,
    create_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ttnn_operation_name_idx      ON ttnn_ops_v5.ttnn_operation(operation_name);
CREATE INDEX ttnn_operation_base_name_idx ON ttnn_ops_v5.ttnn_operation(base_operation_name);

-- ---------------------------------------------------------------------------
-- 2. ttnn_model
--    Each row is a real, named model. No synthetic "multiple" entries.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_model (
    ttnn_model_id       SERIAL PRIMARY KEY,
    source_file         TEXT NOT NULL,
    hf_model_identifier TEXT,
    model_family        TEXT,
    model_name          TEXT UNIQUE,   -- short lowercase id; see derive_model_name() in loader
    create_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts           TIMESTAMPTZ,
    UNIQUE (source_file, hf_model_identifier)
);
CREATE INDEX ttnn_model_source_file_idx ON ttnn_ops_v5.ttnn_model(source_file);
CREATE INDEX ttnn_model_family_idx      ON ttnn_ops_v5.ttnn_model(model_family);

-- ---------------------------------------------------------------------------
-- 3. ttnn_hardware
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_hardware (
    ttnn_hardware_id    SERIAL PRIMARY KEY,
    board_type          TEXT NOT NULL,
    device_series       TEXT NOT NULL,
    card_count          INTEGER NOT NULL DEFAULT 1,
    create_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (board_type, device_series, card_count)
);

-- ---------------------------------------------------------------------------
-- 4. ttnn_mesh_config
--    Mesh shape only; per-tensor placement is stored in ttnn_configuration.full_config_json.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_mesh_config (
    ttnn_mesh_config_id SERIAL PRIMARY KEY,
    mesh_shape          INTEGER[] NOT NULL,
    device_count        INTEGER NOT NULL,
    create_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (mesh_shape, device_count)
);
CREATE INDEX ttnn_mesh_config_shape_idx ON ttnn_ops_v5.ttnn_mesh_config(mesh_shape);

INSERT INTO ttnn_ops_v5.ttnn_mesh_config (mesh_shape, device_count) VALUES ('{1,1}', 1);

-- ---------------------------------------------------------------------------
-- 5. ttnn_configuration
--    One row per unique (op + args + hardware + mesh) combination.
--    config_hash is the canonical identity key.
--    full_config_json is the source of truth for arguments.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_configuration (
    ttnn_configuration_id   BIGSERIAL PRIMARY KEY,
    operation_id            INTEGER NOT NULL
                                REFERENCES ttnn_ops_v5.ttnn_operation(ttnn_operation_id),
    hardware_id             INTEGER
                                REFERENCES ttnn_ops_v5.ttnn_hardware(ttnn_hardware_id),
    mesh_config_id          INTEGER
                                REFERENCES ttnn_ops_v5.ttnn_mesh_config(ttnn_mesh_config_id),
    config_hash             TEXT NOT NULL UNIQUE,
    full_config_json        JSONB NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'observed'
                                CHECK (status IN ('observed', 'expected', 'validated',
                                                  'deprecated', 'removed')),
    first_seen_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_ts            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ttnn_configuration_operation_id_idx    ON ttnn_ops_v5.ttnn_configuration(operation_id);
CREATE INDEX ttnn_configuration_hardware_id_idx     ON ttnn_ops_v5.ttnn_configuration(hardware_id);
CREATE INDEX ttnn_configuration_mesh_config_id_idx  ON ttnn_ops_v5.ttnn_configuration(mesh_config_id);
CREATE INDEX ttnn_configuration_status_idx          ON ttnn_ops_v5.ttnn_configuration(status);

-- ---------------------------------------------------------------------------
-- 6. ttnn_configuration_model
--    Which real models used each configuration.
--    Models are always enumerated here — no synthetic aggregates.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.ttnn_configuration_model (
    configuration_id    BIGINT NOT NULL
                            REFERENCES ttnn_ops_v5.ttnn_configuration(ttnn_configuration_id),
    model_id            INTEGER NOT NULL
                            REFERENCES ttnn_ops_v5.ttnn_model(ttnn_model_id),
    execution_count     INTEGER NOT NULL DEFAULT 1,
    first_seen_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (configuration_id, model_id)
);
CREATE INDEX ttnn_config_model_model_id_idx ON ttnn_ops_v5.ttnn_configuration_model(model_id);

-- ---------------------------------------------------------------------------
-- 7. trace_run
--    One row per load event (one JSON file load / one model trace session).
--    NO model_id: which models are in a trace is always derived from
--    trace_run_config → ttnn_configuration_model → ttnn_model.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.trace_run (
    trace_run_id    SERIAL PRIMARY KEY,
    hardware_id     INTEGER
                        REFERENCES ttnn_ops_v5.ttnn_hardware(ttnn_hardware_id),
    tt_metal_sha    TEXT,
    traced_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config_count    INTEGER,    -- denormalized, updated after load
    notes           TEXT
);
CREATE INDEX trace_run_hardware_id_idx  ON ttnn_ops_v5.trace_run(hardware_id);
CREATE INDEX trace_run_traced_at_idx    ON ttnn_ops_v5.trace_run(traced_at DESC);
CREATE INDEX trace_run_sha_idx          ON ttnn_ops_v5.trace_run(tt_metal_sha)
    WHERE tt_metal_sha IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 8. trace_run_config
--    Which configurations belong to which trace (snapshot / load-event view).
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.trace_run_config (
    trace_run_id        INTEGER NOT NULL
                            REFERENCES ttnn_ops_v5.trace_run(trace_run_id),
    configuration_id    BIGINT NOT NULL
                            REFERENCES ttnn_ops_v5.ttnn_configuration(ttnn_configuration_id),
    execution_count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (trace_run_id, configuration_id)
);
CREATE INDEX trace_run_config_configuration_id_idx
    ON ttnn_ops_v5.trace_run_config(configuration_id);

-- ---------------------------------------------------------------------------
-- 9. trace_run_model  (derived / materialized view of models per trace)
--     Populated by the loader after inserting trace_run_config rows.
--     Answers "which models are in trace N?" without a multi-join.
-- ---------------------------------------------------------------------------
CREATE TABLE ttnn_ops_v5.trace_run_model (
    trace_run_id    INTEGER NOT NULL
                        REFERENCES ttnn_ops_v5.trace_run(trace_run_id),
    model_id        INTEGER NOT NULL
                        REFERENCES ttnn_ops_v5.ttnn_model(ttnn_model_id),
    PRIMARY KEY (trace_run_id, model_id)
);
CREATE INDEX trace_run_model_model_id_idx ON ttnn_ops_v5.trace_run_model(model_id);

-- ---------------------------------------------------------------------------
-- Verify
-- ---------------------------------------------------------------------------
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'ttnn_ops_v5'
ORDER BY table_name;
