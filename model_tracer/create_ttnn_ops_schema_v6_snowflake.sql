-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
--
-- SPDX-License-Identifier: Apache-2.0

-- TTNN Ops Database Schema v6 — SNOWFLAKE edition
--
-- Snowflake port of model_tracer/destructively_create_ttnn_ops_schema_v6.sql.
-- Creates SELF_SERVE.TTNN_OPS_V6, the schema read/written by
-- tests/sweep_framework/load_ttnn_ops_data_v2.py (Snowflake-only).
--
-- ⚠️  DESTRUCTIVE: this DROPs the target schema and all its data, then recreates
--     it empty. To build a NON-destructive copy (e.g. to test without touching
--     the live TTNN_OPS_V6), find/replace TTNN_OPS_V6 -> TTNN_OPS_V7 below.
--
-- Postgres -> Snowflake differences (vs the .sql this is ported from):
--   * SERIAL / BIGSERIAL PRIMARY KEY -> plain NUMBER id columns. Ids are
--     allocated from Snowflake SEQUENCEs that the loader creates on demand
--     (_ensure_sequences seeds SEQ_<TABLE> at MAX(id)+1), so no sequences are
--     created here — that keeps them correctly seeded past the mesh seed row.
--   * full_config_json JSONB -> VARCHAR. It stores Postgres' exact jsonb text so
--     JSON numbers (1.0 vs 1, float formatting) round-trip identically; use
--     PARSE_JSON(full_config_json) at query time for semi-structured access.
--   * mesh_shape INTEGER[] -> ARRAY (stored via PARSE_JSON of a JSON array).
--   * base_operation_name: Postgres GENERATED ALWAYS column has no Snowflake
--     equivalent — it is a plain VARCHAR the loader computes and inserts.
--   * CHECK constraints and (partial) INDEXes are unsupported / not applicable
--     (Snowflake uses micro-partitions; add CLUSTER BY if ever needed) — omitted.
--   * PRIMARY KEY / UNIQUE / FOREIGN KEY are included as metadata only. Snowflake
--     does NOT enforce them (only NOT NULL is enforced); they document intent and
--     inform tooling/optimizer.
--   * DEFAULT NOW() -> DEFAULT CURRENT_TIMESTAMP().

DROP SCHEMA IF EXISTS SELF_SERVE.TTNN_OPS_V6;
CREATE SCHEMA SELF_SERVE.TTNN_OPS_V6;

-- ---------------------------------------------------------------------------
-- 1. ttnn_operation
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_OPERATION (
    TTNN_OPERATION_ID   NUMBER(38,0) NOT NULL,
    OPERATION_NAME      VARCHAR      NOT NULL,
    -- Postgres GENERATED column; populated by the loader (_base_operation_name).
    BASE_OPERATION_NAME VARCHAR,
    CREATE_TS           TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (TTNN_OPERATION_ID),
    UNIQUE (OPERATION_NAME)
);

-- ---------------------------------------------------------------------------
-- 2. ttnn_model
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_MODEL (
    TTNN_MODEL_ID       NUMBER(38,0) NOT NULL,
    SOURCE_FILE         VARCHAR      NOT NULL,
    HF_MODEL_IDENTIFIER VARCHAR,
    MODEL_FAMILY        VARCHAR,
    MODEL_NAME          VARCHAR,
    CREATE_TS           TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    UPDATE_TS           TIMESTAMP_TZ,
    PRIMARY KEY (TTNN_MODEL_ID),
    UNIQUE (MODEL_NAME),
    UNIQUE (SOURCE_FILE, HF_MODEL_IDENTIFIER)
);

-- ---------------------------------------------------------------------------
-- 3. ttnn_hardware
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_HARDWARE (
    TTNN_HARDWARE_ID NUMBER(38,0) NOT NULL,
    BOARD_TYPE       VARCHAR      NOT NULL,
    DEVICE_SERIES    VARCHAR      NOT NULL,
    CARD_COUNT       NUMBER(38,0) NOT NULL DEFAULT 1,
    CREATE_TS        TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (TTNN_HARDWARE_ID),
    UNIQUE (BOARD_TYPE, DEVICE_SERIES, CARD_COUNT)
);

-- ---------------------------------------------------------------------------
-- 4. ttnn_mesh_config  (mesh_shape stored as ARRAY)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_MESH_CONFIG (
    TTNN_MESH_CONFIG_ID NUMBER(38,0) NOT NULL,
    MESH_SHAPE          ARRAY        NOT NULL,
    DEVICE_COUNT        NUMBER(38,0) NOT NULL,
    CREATE_TS           TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (TTNN_MESH_CONFIG_ID),
    UNIQUE (MESH_SHAPE, DEVICE_COUNT)
);

-- Canonical seed row (matches the loader's PARSE_JSON insert form so
-- get_or_create_mesh_config finds it instead of creating a duplicate).
INSERT INTO SELF_SERVE.TTNN_OPS_V6.TTNN_MESH_CONFIG
    (TTNN_MESH_CONFIG_ID, MESH_SHAPE, DEVICE_COUNT, CREATE_TS)
    SELECT 1, PARSE_JSON('[1,1]'), 1, CURRENT_TIMESTAMP();

-- ---------------------------------------------------------------------------
-- 5. ttnn_configuration  (full_config_json is the source of truth, stored as text)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_CONFIGURATION (
    TTNN_CONFIGURATION_ID NUMBER(38,0) NOT NULL,
    OPERATION_ID          NUMBER(38,0) NOT NULL,
    HARDWARE_ID           NUMBER(38,0),
    MESH_CONFIG_ID        NUMBER(38,0),
    CONFIG_HASH           VARCHAR      NOT NULL,
    FULL_CONFIG_JSON      VARCHAR      NOT NULL,
    -- Allowed values: observed | expected | validated | deprecated | removed
    -- (Postgres CHECK is not supported in Snowflake; enforced by the loader.)
    STATUS                VARCHAR      NOT NULL DEFAULT 'observed',
    FIRST_SEEN_TS         TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    LAST_SEEN_TS          TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (TTNN_CONFIGURATION_ID),
    UNIQUE (CONFIG_HASH),
    FOREIGN KEY (OPERATION_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_OPERATION (TTNN_OPERATION_ID),
    FOREIGN KEY (HARDWARE_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_HARDWARE (TTNN_HARDWARE_ID),
    FOREIGN KEY (MESH_CONFIG_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_MESH_CONFIG (TTNN_MESH_CONFIG_ID)
);

-- ---------------------------------------------------------------------------
-- 6. ttnn_configuration_model  (derived aggregate: counts per config+model)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TTNN_CONFIGURATION_MODEL (
    CONFIGURATION_ID NUMBER(38,0) NOT NULL,
    MODEL_ID         NUMBER(38,0) NOT NULL,
    EXECUTION_COUNT  NUMBER(38,0) NOT NULL DEFAULT 1,
    FIRST_SEEN_TS    TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    LAST_SEEN_TS     TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (CONFIGURATION_ID, MODEL_ID),
    FOREIGN KEY (CONFIGURATION_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_CONFIGURATION (TTNN_CONFIGURATION_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_MODEL (TTNN_MODEL_ID)
);

-- ---------------------------------------------------------------------------
-- 7. trace_run  (one row per imported trace artifact)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TRACE_RUN (
    TRACE_RUN_ID NUMBER(38,0) NOT NULL,
    TRACE_UID    VARCHAR      NOT NULL,
    HARDWARE_ID  NUMBER(38,0),
    TT_METAL_SHA VARCHAR,
    TRACED_AT    TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    CONFIG_COUNT NUMBER(38,0),
    NOTES        VARCHAR,
    PYTEST_ARGS  VARCHAR,
    TT_KMD       VARCHAR,
    TT_SMI       VARCHAR,
    TT_FIRMWARE  VARCHAR,
    PRIMARY KEY (TRACE_RUN_ID),
    UNIQUE (TRACE_UID),
    FOREIGN KEY (HARDWARE_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_HARDWARE (TTNN_HARDWARE_ID)
);

-- ---------------------------------------------------------------------------
-- 8. trace_run_configuration_model  (canonical per-trace per-config per-model counts)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TRACE_RUN_CONFIGURATION_MODEL (
    TRACE_RUN_ID     NUMBER(38,0) NOT NULL,
    CONFIGURATION_ID NUMBER(38,0) NOT NULL,
    MODEL_ID         NUMBER(38,0) NOT NULL,
    EXECUTION_COUNT  NUMBER(38,0) NOT NULL DEFAULT 1,
    FIRST_SEEN_TS    TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    LAST_SEEN_TS     TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (TRACE_RUN_ID, CONFIGURATION_ID, MODEL_ID),
    FOREIGN KEY (TRACE_RUN_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TRACE_RUN (TRACE_RUN_ID),
    FOREIGN KEY (CONFIGURATION_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_CONFIGURATION (TTNN_CONFIGURATION_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_MODEL (TTNN_MODEL_ID)
);

-- ---------------------------------------------------------------------------
-- 9. trace_run_config  (derived aggregate: counts per trace+config)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TRACE_RUN_CONFIG (
    TRACE_RUN_ID     NUMBER(38,0) NOT NULL,
    CONFIGURATION_ID NUMBER(38,0) NOT NULL,
    EXECUTION_COUNT  NUMBER(38,0) NOT NULL DEFAULT 1,
    PRIMARY KEY (TRACE_RUN_ID, CONFIGURATION_ID),
    FOREIGN KEY (TRACE_RUN_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TRACE_RUN (TRACE_RUN_ID),
    FOREIGN KEY (CONFIGURATION_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_CONFIGURATION (TTNN_CONFIGURATION_ID)
);

-- ---------------------------------------------------------------------------
-- 10. trace_run_model  (derived: models per trace)
-- ---------------------------------------------------------------------------
CREATE TABLE SELF_SERVE.TTNN_OPS_V6.TRACE_RUN_MODEL (
    TRACE_RUN_ID NUMBER(38,0) NOT NULL,
    MODEL_ID     NUMBER(38,0) NOT NULL,
    PRIMARY KEY (TRACE_RUN_ID, MODEL_ID),
    FOREIGN KEY (TRACE_RUN_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TRACE_RUN (TRACE_RUN_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES SELF_SERVE.TTNN_OPS_V6.TTNN_MODEL (TTNN_MODEL_ID)
);
