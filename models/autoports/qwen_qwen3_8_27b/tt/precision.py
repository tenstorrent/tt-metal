# SPDX-License-Identifier: Apache-2.0
"""Versioned full-model precision contract, shared by every generator constructor."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    'schema_version': 1, 'config_id': 'baseline_bfp4_lofi',
    'weight_groups': {'attention': 'bfloat4_b', 'mlp': 'bfloat4_b', 'down': 'bfloat4_b',
                      'embedding': 'bfloat16', 'head': 'bfloat16', 'norm': 'bfloat16'},
    'layer_exceptions': {},
    'compute_fidelities': {'projection': 'LoFi', 'fused_prefill': 'LoFi',
                           'fused_decode': 'LoFi', 'sdpa': 'LoFi', 'head': 'HiFi4', 'norm': 'HiFi4'},
    'activation_dtype': {'ordinary_attention': 'bfloat16', 'ordinary_mlp': 'bfloat16',
                         'prefill_column': 'bfloat8_b', 'fused_row': 'bfloat16'},
    'residual_dtype': 'bfloat16',
    'ccl_dtype': {'decode_b1': 'bfloat8_b', 'prefill': 'bfloat8_b', 'native_batched': 'bfloat16'},
    'kv_cache_dtype': 'bfloat8_b',
    'logits_sampling': {'logits': 'bfloat16', 'sampling': 'bfloat16', 'token_ids': 'uint32'},
    'recurrent_state_dtype': 'float32', 'max_context': 262144,
}


def load_precision(value=None):
    """Explicit override > environment > source-owned selected policy."""
    if value is None:
        value = os.environ.get('QWEN38_PRECISION_CONFIG')
    if value is None:
        value = ROOT / 'tt/selected_precision_config.json'
    config = copy.deepcopy(BASELINE) if value is None else (
        copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text()))
    if set(config) != set(BASELINE):
        raise ValueError('Precision configuration must contain exactly the versioned contract fields')
    for field in ('weight_groups', 'compute_fidelities', 'activation_dtype', 'ccl_dtype', 'logits_sampling'):
        if set(config[field]) != set(BASELINE[field]):
            raise ValueError(f'Incomplete or unknown precision keys in {field}')
    # These are enforced runtime invariants, not silently ignored options. The
    # recurrence, shared workspace and common sampler require these representations.
    for field in ('schema_version', 'residual_dtype', 'logits_sampling', 'recurrent_state_dtype', 'max_context'):
        if config[field] != BASELINE[field]:
            raise ValueError(f'Unsupported precision invariant {field}: {config[field]}')
    for field, keys in {'weight_groups': ('embedding', 'norm'), 'compute_fidelities': ('norm',),
                        'ccl_dtype': ('prefill', 'native_batched'), 'activation_dtype': ('fused_row',)}.items():
        if set(config[field]) != set(BASELINE[field]):
            raise ValueError(f'Incomplete precision field {field}')
        for key in keys:
            if config[field][key] != BASELINE[field][key]:
                raise ValueError(f'Fixed runtime representation {field}.{key}')
    for dtype in [*config['weight_groups'].values(), *config['activation_dtype'].values(),
                  *config['ccl_dtype'].values(), config['kv_cache_dtype']]:
        if dtype not in ('bfloat16', 'bfloat8_b', 'bfloat4_b'):
            raise ValueError(f'Unsupported dtype {dtype}')
    if any(f not in ('LoFi', 'HiFi2', 'HiFi4') for f in config['compute_fidelities'].values()):
        raise ValueError('Unsupported math fidelity')
    for layer, override in config['layer_exceptions'].items():
        if not isinstance(layer, str) or not layer.isdigit() or not 0 <= int(layer) < 64:
            raise ValueError('Invalid layer exception')
        if set(override) - {'attention_dtype', 'mlp_dtype', 'down_dtype', 'fidelity'}:
            raise ValueError('Unsupported layer override')
        for key, value in override.items():
            allowed = ('LoFi', 'HiFi2', 'HiFi4') if key == 'fidelity' else ('bfloat16', 'bfloat8_b', 'bfloat4_b')
            if value not in allowed:
                raise ValueError(f'Unsupported layer override {layer}.{key}: {value}')
    return config


def decoder_policy(config, index):
    weights, fidelity = config['weight_groups'], config['compute_fidelities']
    policy = {g + '_dtype': weights[g] for g in ('attention', 'mlp', 'down')}
    policy.update(fidelity=fidelity['projection'], fused_fidelity=fidelity['fused_decode'],
                  fused_prefill_fidelity=fidelity['fused_prefill'], sdpa_fidelity=fidelity['sdpa'],
                  kv_dtype=config['kv_cache_dtype'], ccl_dtype=config['ccl_dtype']['decode_b1'],
                  attention_activation=config['activation_dtype']['ordinary_attention'],
                  mlp_activation=config['activation_dtype']['ordinary_mlp'],
                  precision_prefill_activation=config['activation_dtype']['prefill_column'])
    policy.update(config['layer_exceptions'].get(str(index), {}))
    grouped = os.environ.get('QWEN38_GROUPED_RECURRENCE', '0')
    if grouped not in ('0', '1'):
        raise ValueError('QWEN38_GROUPED_RECURRENCE must be 0 or 1')
    if grouped == '1':
        policy['grouped_recurrence'] = True
    masked = os.environ.get('QWEN38_MASKED_RECURRENCE', '0')
    if masked not in ('0', '1'):
        raise ValueError('QWEN38_MASKED_RECURRENCE must be 0 or 1')
    if masked == '1':
        policy['masked_recurrence'] = True
    flat_prefill = os.environ.get('QWEN38_FLAT_PREFILL', '0')
    if flat_prefill not in ('0', '1'):
        raise ValueError('QWEN38_FLAT_PREFILL must be 0 or 1')
    if flat_prefill == '1':
        policy['flat_prefill'] = True
    return policy
