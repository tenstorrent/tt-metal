-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
--
-- SPDX-License-Identifier: Apache-2.0

-- TTNN Ops Database Schema v2
--
-- Key V2 Changes:
--   1. Per-tensor placement: Each argument stores its own tensor_placement_json
--   2. ttnn_mesh_config simplified: Only mesh_shape and device_count (no placement)
--   3. ttnn_argument enhanced: Added tensor_placement_json column for per-tensor placement
--   4. Arguments format: Dictionary {arg0: ..., arg1: ...} instead of array
--
-- This schema supports the current tracer output format where each tensor can have
-- different placement strategies (e.g., arg0 replicated, arg1 sharded on different dims)
--
-- Drop and recreate schema
DROP SCHEMA IF EXISTS ttnn_ops_v2 CASCADE;
CREATE SCHEMA ttnn_ops_v2;

-- 1. ttnn_operation
CREATE TABLE ttnn_ops_v2.ttnn_operation (
    ttnn_operation_id SERIAL PRIMARY KEY,
    operation_name TEXT NOT NULL UNIQUE,
    base_operation_name TEXT GENERATED ALWAYS AS (
        REGEXP_REPLACE(operation_name,
            '^(ttnn::|experimental::|ttnn::experimental::|ttnn::transformer::)', '')
    ) STORED,
    operation_type TEXT CHECK (operation_type IN ('unary', 'binary', 'ternary', 'multi_input', 'special')),
    tensor_input_count INTEGER,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ttnn_operation_name_idx ON ttnn_ops_v2.ttnn_operation(operation_name);
CREATE INDEX ttnn_operation_base_name_idx ON ttnn_ops_v2.ttnn_operation(base_operation_name);
CREATE INDEX ttnn_operation_type_idx ON ttnn_ops_v2.ttnn_operation(operation_type);

-- 2. ttnn_model
CREATE TABLE ttnn_ops_v2.ttnn_model (
    ttnn_model_id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    hf_model_identifier TEXT,
    model_family TEXT,
    task_type TEXT,
    is_lead_model BOOLEAN NOT NULL DEFAULT false,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_file, hf_model_identifier)
);
CREATE INDEX ttnn_model_source_file_idx ON ttnn_ops_v2.ttnn_model(source_file);
CREATE INDEX ttnn_model_hf_identifier_idx ON ttnn_ops_v2.ttnn_model(hf_model_identifier);
CREATE INDEX ttnn_model_family_idx ON ttnn_ops_v2.ttnn_model(model_family);
CREATE INDEX ttnn_model_lead_idx ON ttnn_ops_v2.ttnn_model(is_lead_model) WHERE is_lead_model = true;

-- 3. ttnn_hardware
CREATE TABLE ttnn_ops_v2.ttnn_hardware (
    ttnn_hardware_id SERIAL PRIMARY KEY,
    board_type TEXT NOT NULL,
    device_series TEXT NOT NULL,
    card_count INTEGER NOT NULL,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (board_type, device_series, card_count)
);

-- 4. ttnn_mesh_config (for multi-device configurations)
-- Note: Placement is now stored per-tensor in ttnn_argument.tensor_placement_json
CREATE TABLE ttnn_ops_v2.ttnn_mesh_config (
    ttnn_mesh_config_id SERIAL PRIMARY KEY,
    mesh_shape INTEGER[] NOT NULL,
    device_count INTEGER NOT NULL,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (mesh_shape, device_count)
);
CREATE INDEX ttnn_mesh_config_shape_idx ON ttnn_ops_v2.ttnn_mesh_config(mesh_shape);
CREATE INDEX ttnn_mesh_config_device_count_idx ON ttnn_ops_v2.ttnn_mesh_config(device_count);

-- Insert default 1x1 single-device mesh
INSERT INTO ttnn_ops_v2.ttnn_mesh_config (mesh_shape, device_count)
VALUES ('{1,1}', 1);

-- 5. ttnn_configuration (mesh_config_id included - mesh affects test execution)
CREATE TABLE ttnn_ops_v2.ttnn_configuration (
    ttnn_configuration_id SERIAL PRIMARY KEY,
    operation_id INTEGER NOT NULL REFERENCES ttnn_ops_v2.ttnn_operation(ttnn_operation_id),
    hardware_id INTEGER REFERENCES ttnn_ops_v2.ttnn_hardware(ttnn_hardware_id),
    mesh_config_id INTEGER REFERENCES ttnn_ops_v2.ttnn_mesh_config(ttnn_mesh_config_id),
    primary_dtype TEXT,
    primary_storage_type TEXT,
    primary_layout TEXT,
    primary_memory_layout TEXT,
    primary_buffer_type TEXT,
    primary_shape INTEGER[],
    config_hash TEXT NOT NULL UNIQUE,
    full_config_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'observed' CHECK (status IN ('observed', 'expected', 'validated', 'deprecated', 'removed')),
    first_seen_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    removed_ts TIMESTAMPTZ,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ttnn_configuration_operation_id_idx ON ttnn_ops_v2.ttnn_configuration(operation_id);
CREATE INDEX ttnn_configuration_hardware_id_idx ON ttnn_ops_v2.ttnn_configuration(hardware_id);
CREATE INDEX ttnn_configuration_mesh_config_id_idx ON ttnn_ops_v2.ttnn_configuration(mesh_config_id);
CREATE INDEX ttnn_configuration_dtype_idx ON ttnn_ops_v2.ttnn_configuration(primary_dtype);
CREATE INDEX ttnn_configuration_memory_layout_idx ON ttnn_ops_v2.ttnn_configuration(primary_memory_layout);
CREATE INDEX ttnn_configuration_status_idx ON ttnn_ops_v2.ttnn_configuration(status);

-- 6. ttnn_argument (with per-tensor placement support)
CREATE TABLE ttnn_ops_v2.ttnn_argument (
    ttnn_argument_id SERIAL PRIMARY KEY,
    configuration_id INTEGER NOT NULL REFERENCES ttnn_ops_v2.ttnn_configuration(ttnn_configuration_id),
    arg_index INTEGER NOT NULL,
    arg_name TEXT NOT NULL,
    arg_semantic_type TEXT CHECK (arg_semantic_type IN (
        'input_tensor', 'weight_tensor', 'bias_tensor', 'output_preallocated',
        'memory_config', 'scalar_param', 'dimension_param', 'shape_param', 'compute_config'
    )),
    is_tensor BOOLEAN NOT NULL,
    is_tensor_list BOOLEAN NOT NULL DEFAULT false,
    tensor_count INTEGER,
    tensor_dtype TEXT,
    tensor_storage_type TEXT,
    tensor_layout TEXT,
    tensor_memory_layout TEXT,
    tensor_buffer_type TEXT,
    tensor_shape INTEGER[],
    shard_shape INTEGER[],
    shard_orientation TEXT,
    core_grid_x INTEGER,
    core_grid_y INTEGER,
    -- Per-tensor placement information (for multi-device tensors)
    tensor_placement_json JSONB,  -- Stores: {placement: "['PlacementShard(2)']", distribution_shape: "[1,2]", mesh_device_shape: "[1,2]"}
    tensor_spec_json JSONB,       -- Full tensor specification
    scalar_value_json JSONB,
    unsupported_type_string TEXT,
    create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (configuration_id, arg_index)
);
CREATE INDEX ttnn_argument_configuration_id_idx ON ttnn_ops_v2.ttnn_argument(configuration_id);
CREATE INDEX ttnn_argument_is_tensor_idx ON ttnn_ops_v2.ttnn_argument(is_tensor);
CREATE INDEX ttnn_argument_tensor_dtype_idx ON ttnn_ops_v2.ttnn_argument(tensor_dtype) WHERE is_tensor = true OR is_tensor_list = true;
CREATE INDEX ttnn_argument_semantic_type_idx ON ttnn_ops_v2.ttnn_argument(arg_semantic_type);
CREATE INDEX ttnn_argument_shard_shape_idx ON ttnn_ops_v2.ttnn_argument(shard_shape) WHERE shard_shape IS NOT NULL;
CREATE INDEX ttnn_argument_tensor_placement_idx ON ttnn_ops_v2.ttnn_argument USING GIN (tensor_placement_json) WHERE tensor_placement_json IS NOT NULL;

-- 7. ttnn_configuration_model (junction table: config <-> model with execution tracking)
-- Maps the "executions" array from JSON - each execution is a config observed in a specific model
CREATE TABLE ttnn_ops_v2.ttnn_configuration_model (
    configuration_id INTEGER NOT NULL REFERENCES ttnn_ops_v2.ttnn_configuration(ttnn_configuration_id),
    model_id INTEGER NOT NULL REFERENCES ttnn_ops_v2.ttnn_model(ttnn_model_id),
    execution_count INTEGER NOT NULL DEFAULT 1,  -- How many times this config executed in this model (from JSON "count" field)
    first_seen_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (configuration_id, model_id)
);
CREATE INDEX ttnn_config_model_model_id_idx ON ttnn_ops_v2.ttnn_configuration_model(model_id);
CREATE INDEX ttnn_config_model_execution_count_idx ON ttnn_ops_v2.ttnn_configuration_model(execution_count);

-- Verify schema created
SELECT table_name FROM information_schema.tables WHERE table_schema = 'ttnn_ops_v2' ORDER BY table_name;
