# SPDX-License-Identifier: Apache-2.0
"""Owned CPU-only BF16 HF reference with lazy, existing-shard weight loading.

Default prepare mode reads headers only. Validation/target modes require a
<=35GB cgroup and no TT device nodes; they never copy checkpoint weight files.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import time

MEMORY_LIMIT = 35_000_000_000
MODEL = 'Qwen/Qwen3.8-27B'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def build_index(checkpoint):
    """Read safetensors headers, never get_tensor or copy payloads."""
    root = Path(checkpoint).resolve()
    weight_map = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    result, shapes, total = {}, {}, 0
    for original, filename in weight_map.items():
        if not (original.startswith('model.language_model.') or original == 'lm_head.weight'):
            continue
        path = (root/filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Checkpoint shard escapes checkpoint directory')
        if filename not in headers:
            with path.open('rb') as f:
                length_bytes = f.read(8)
                if len(length_bytes) != 8:
                    raise ValueError('Truncated safetensors header')
                length = struct.unpack('<Q', length_bytes)[0]
                if not 0 < length < 32 * 1024 * 1024:
                    raise ValueError('Unexpected safetensors header size')
                raw = f.read(length)
                if len(raw) != length:
                    raise ValueError('Truncated safetensors header')
            headers[filename] = (json.loads(raw), 8 + length, path.stat().st_size)
        header, start, size = headers[filename]
        info = header[original]
        begin, end = info['data_offsets']
        if info['dtype'] != 'BF16':
            raise ValueError('Reference text weights must remain exact BF16')
        if begin < 0 or end - begin != math.prod(info['shape']) * 2 or start + end > size:
            raise ValueError('Invalid/truncated weight range')
        name = original.replace('model.language_model.', 'model.', 1)
        if name in result:
            raise ValueError('Duplicate mapped parameter')
        result[name] = {'safetensors_file': str(path), 'weight_name': original}
        shapes[name] = info['shape']
        total += end - begin
    if not result:
        raise ValueError('No text parameters in checkpoint index')
    return result, {'weight_bytes': total, 'parameter_shapes': shapes,
                    'shard_count': len(headers), 'weight_copy_bytes': 0}


def checkpoint_identity(root):
    return {name: sha(Path(root)/name) for name in
            ['config.json', 'generation_config.json', 'model.safetensors.index.json']}


def check_guard(cgroup=Path('/sys/fs/cgroup'), device_nodes=None):
    cgroup = Path(cgroup)
    limit = (cgroup/'memory.max').read_text().strip()
    if limit == 'max' or not 0 < int(limit) <= MEMORY_LIMIT:
        raise RuntimeError('CPU reference requires a cgroup memory.max <=35000000000')
    nodes = list(Path('/dev').glob('tenstorrent*')) if device_nodes is None else device_nodes
    if nodes:
        raise RuntimeError('CPU reference container must expose no TT devices')
    return {'memory_max_bytes': int(limit), 'tt_device_nodes': []}


def require_validation(receipt, source_hash, identity):
    if (receipt.get('status') != 'validated' or receipt.get('top100_exact') is not True
            or receipt.get('source_sha256') != source_hash
            or receipt.get('checkpoint_identity') != identity):
        raise ValueError('Target requires passing source/checkpoint-bound validation receipt')


def load_streamed_model(root, index_dir, index, metadata):
    import torch
    from accelerate import disk_offload
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding)

    config = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    if config.num_hidden_layers != 64 or config.vocab_size != 248320:
        raise ValueError('Expected exact 64-layer Qwen3.8-27B text architecture')
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    # Accelerate casts loaded values to existing parameter dtype. Set only
    # parameter metadata; do not downcast native rotary/nonpersistent buffers.
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            module._parameters[name] = torch.nn.Parameter(param.to(dtype=torch.bfloat16),
                                                         requires_grad=False)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    params = dict(model.named_parameters())
    if set(params) != set(index):
        raise ValueError('Checkpoint parameter mapping differs from native text model')
    for name, param in params.items():
        if list(param.shape) != metadata['parameter_shapes'][name] or param.dtype != torch.bfloat16:
            raise ValueError('Parameter shape/dtype mismatch: ' + name)
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError('Unmaterialized native buffer')
    # An existing index prevents disk_offload from creating .dat weight copies.
    write_json(index_dir/'index.json', index)
    model = disk_offload(model, str(index_dir), execution_device=torch.device('cpu'),
                         offload_buffers=False,
                         preload_module_classes=['Qwen3_5DecoderLayer'])
    return model.eval()


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = {'status': 'preparing', 'mode': args.mode, 'model': MODEL,
              'source_sha256': sha(__file__), 'device_hours': 0,
              'checkpoint_identity': checkpoint_identity(args.checkpoint),
              'weight_copy_bytes': 0, 'scored': False}
    write_json(output/'receipt.json', record)
    try:
        index, metadata = build_index(args.checkpoint)
        record['metadata'] = {k:v for k,v in metadata.items() if k != 'parameter_shapes'}
        if args.mode == 'prepare':
            write_json(output/'index.json', index)
            record['status'] = 'prepared_headers_only'
            return record
        if not args.owner or os.environ.get('QWEN38_HF_CPU_OWNER') != args.owner:
            raise RuntimeError('Explicit owned CPU container identity required')
        record['owner'] = args.owner
        record['guard'] = check_guard()
        import torch
        from safetensors.torch import save_file
        torch.set_num_threads(args.threads)
        if args.mode == 'validate':
            if args.reference_refpt is None:
                raise ValueError('Validation requires retained native .refpt reference')
            reference = torch.load(args.reference_refpt, map_location='cpu', weights_only=True)
            if reference['hf_model_id'] != MODEL:
                raise ValueError('Reference model identity mismatch')
            entry = reference['entries'][0]
            ids = entry['prompt_tokens'].reshape(-1).tolist()
            expected = entry['topk_tokens'][0].reshape(-1).tolist()
            if len(ids) != 203 or len(expected) != 100:
                raise ValueError('Expected cached 203-token native AIME first-step reference')
            record['reference_sha256'] = sha(args.reference_refpt)
            record['reference_scope'] = 'First native prefill step; retained top100 IDs/ranks only, no retained logit values'
        else:
            if args.validation_receipt is None or args.input_ids_json is None:
                raise ValueError('Target requires validation receipt and frozen input IDs')
            require_validation(json.loads(args.validation_receipt.read_text()),
                               record['source_sha256'], record['checkpoint_identity'])
            payload = json.loads(args.input_ids_json.read_text())
            if payload.get('prompt_mode') != 'native_chat_once' or not payload.get('provenance'):
                raise ValueError('Frozen history needs native-chat provenance')
            ids = payload['input_ids']
            if not 6000 <= len(ids) <= 8192:
                raise ValueError('Approved diagnostic length range is 6000..8192')
            record['input_artifact_sha256'] = sha(args.input_ids_json)
            record['provenance'] = payload['provenance']
        if any(type(token) is not int or not 0 <= token < 248320 for token in ids):
            raise ValueError('Invalid native token IDs')
        record['input_tokens'] = len(ids)
        record['input_ids_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        record['status'] = 'loading'
        write_json(output/'receipt.json', record)
        index_dir = output/'offload_index'
        index_dir.mkdir()
        model = load_streamed_model(args.checkpoint, index_dir, index, metadata)
        record['status'] = 'forward'
        record['effective_config'] = {'layers': model.config.num_hidden_layers,
            'vocab_size': model.config.vocab_size, 'attention': model.config._attn_implementation,
            'parameter_dtype': 'bfloat16', 'use_cache': True, 'logits_to_keep': 1,
            'threads': args.threads, 'preload_module_classes': ['Qwen3_5DecoderLayer']}
        write_json(output/'receipt.json', record)
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1].detach().float().contiguous()
        if logits.numel() != 248320 or not torch.isfinite(logits).all():
            raise RuntimeError('Invalid full-vocabulary logits')
        values, indices = logits.topk(100)
        record['top100'] = [{'token':int(t), 'logit':float(v)} for t,v in zip(indices, values)]
        record['logits_origin_dtype'] = str(result.logits.dtype)
        if result.logits.dtype != torch.bfloat16:
            raise RuntimeError('Native reference logits must originate in BF16')
        if args.mode == 'validate':
            record['top100_exact'] = indices.tolist() == expected
            if not record['top100_exact']:
                raise RuntimeError('Retained native top100 IDs/ranks differ; target blocked')
        save_file({'logits': logits}, str(output/'logits.safetensors'))
        record['logits_sha256'] = sha(output/'logits.safetensors')
        record['status'] = 'validated' if args.mode == 'validate' else 'completed'
        return record
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak = Path('/sys/fs/cgroup/memory.peak')
        record['cgroup_memory_peak_bytes'] = int(peak.read_text()) if peak.exists() else None
        write_json(output/'receipt.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare','validate','target'], default='prepare')
    parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/.cache/qwen38-full'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--owner')
    parser.add_argument('--reference-refpt', type=Path)
    parser.add_argument('--input-ids-json', type=Path)
    parser.add_argument('--validation-receipt', type=Path)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('threads must be1..16')
    execute(args)


if __name__ == '__main__': main()
