# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay one native CUDA component input on TT; never an integrated TTS claim."""

import argparse
from copy import deepcopy
import json
import importlib.metadata
import os
from pathlib import Path
import time

import torch

from ..config import load_config
from .manifest import load_manifest, materialize_snapshot, sha256_file
from .metrics import compare_tensors

COMPONENTS = (
    'base_lm.forward', 'residual_lm.forward', 'feat_encoder.forward',
    'fsq_layer.forward', 'feat_decoder.estimator.forward', 'enc_to_lm_proj.forward',
    'fusion_concat_proj.forward', 'lm_to_dit_proj.forward', 'res_to_dit_proj.forward',
    'stop_proj.forward', 'stop_actn.forward', 'stop_head.forward',
    'audio_vae.encode', 'audio_vae.decode',
)


def checkpoint_state(checkpoint, manifest, *, codec=False):
    """Verify content against the oracle before preparing any device weights."""
    for name, record in manifest['checkpoint']['files'].items():
        path = checkpoint / name
        if path.parent.resolve() != checkpoint.resolve() or sha256_file(path) != record['sha256']:
            raise ValueError(f'Checkpoint differs from CUDA oracle: {name}')
    safetensors = checkpoint / ('audiovae.safetensors' if codec else 'model.safetensors')
    if safetensors.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors), device='cpu')
    binary = checkpoint / ('audiovae.pth' if codec else 'pytorch_model.bin')
    if not binary.exists():
        raise FileNotFoundError(f'Missing weights: {safetensors.name} or {binary.name}')
    state = torch.load(binary, map_location='cpu', weights_only=True)
    return state.get('state_dict', state)


def build_component(name, config, state, device, dtype):
    import ttnn
    from ..tt.local import TtLocalDiT, TtLocalEncoder, TtScalarQuantization, make_local_config
    from ..tt.minicpm import TtMiniCPMModel
    from ..tt.ops import TtLinear

    if name.startswith('audio_vae.'):
        from ..tt.audio_vae import AudioVAEConfig, TtAudioVAE
        return TtAudioVAE(state, '', device, dtype, AudioVAEConfig.from_mapping(config.get('audio_vae_config') or {}))
    prefix = name.removesuffix('.forward')
    if prefix in ('base_lm', 'residual_lm'):
        lm = deepcopy(config['lm_config'])
        if prefix == 'residual_lm':
            lm.update(num_hidden_layers=config['residual_lm_num_layers'], vocab_size=0,
                      no_rope=config.get('residual_lm_no_rope', False))
        return TtMiniCPMModel(lm, state, prefix, device, dtype)
    if prefix == 'feat_encoder':
        local_config = make_local_config(config['lm_config'], config['encoder_config'])
        return TtLocalEncoder(local_config, state, prefix, device, dtype, input_dim=config['feat_dim'])
    if prefix == 'feat_decoder.estimator':
        local_config = make_local_config(config['lm_config'], config['dit_config'])
        return TtLocalDiT(local_config, state, prefix, device, dtype, in_channels=config['feat_dim'])
    if prefix == 'fsq_layer':
        return TtScalarQuantization(state, prefix, device, dtype,
                                    scale=config.get('scalar_quantization_scale', 9))
    if prefix == 'stop_actn':
        return ttnn.silu
    return TtLinear(state, prefix, device, dtype)


def replay(args):
    from loguru import logger
    manifest = load_manifest(args.reference)
    events = [e for e in manifest['events'] if e['component'] == args.component]
    if args.event_index < 0 or args.event_index >= len(events):
        raise ValueError(f'{args.component}: event index outside captured range (count={len(events)})')
    event = events[args.event_index]
    inputs = materialize_snapshot(args.reference, manifest, event['args'])
    kwargs = materialize_snapshot(args.reference, manifest, event['kwargs'])
    expected = materialize_snapshot(args.reference, manifest, event['output'])
    # MiniCPM forward also returns the prefill KV list. This first milestone
    # qualifies hidden output only; cache output and forward_step are pending.
    if args.component in ('base_lm.forward', 'residual_lm.forward'):
        expected = expected[0]
    if not isinstance(expected, torch.Tensor):
        raise TypeError('Selected component must return a tensor')
    checkpoint = args.checkpoint.resolve()
    config = load_config(checkpoint)
    codec = args.component.startswith('audio_vae.')
    state = checkpoint_state(checkpoint, manifest, codec=codec)
    import ttnn
    from ..tt.ops import upload
    dtype = {'bfloat16': ttnn.bfloat16, 'float32': ttnn.float32}[args.dtype]
    device = ttnn.open_device(device_id=args.device_id,
                              l1_small_size=getattr(args, 'l1_small_size', 262144))
    try:
        component = build_component(args.component, config, state, device, dtype)
        def transfer(value):
            if isinstance(value, torch.Tensor):
                return upload(value, device, dtype)
            if isinstance(value, tuple):
                return tuple(transfer(x) for x in value)
            if isinstance(value, list):
                return [transfer(x) for x in value]
            if isinstance(value, dict):
                return {k: transfer(v) for k, v in value.items()}
            return value
        if codec:
            # Layout conversion at the actual component input boundary. All
            # activation computation inside the codec remains on TT.
            inputs = list(inputs)
            input_key = 'audio_data' if args.component.endswith('encode') else 'z'
            audio = inputs[0] if inputs else kwargs.pop(input_key)
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            if audio.ndim != 3:
                raise ValueError('Captured codec input must be [batch,channels,time]')
            audio = audio.transpose(1, 2).unsqueeze(1).contiguous()
            if inputs:
                inputs[0] = audio
            else:
                inputs.append(audio)
            # Sample-rate conditioning is scalar metadata. The initial port
            # handles one shared rate for the batch, not per-item rates.
            if 'sr_cond' in kwargs:
                rate = kwargs.pop('sr_cond')
                if rate is not None:
                    flat = rate.flatten()
                    if not torch.all(flat == flat[0]):
                        raise NotImplementedError('Per-item codec sample rates are pending')
                    kwargs['sample_rate'] = int(flat[0])
            component = component.encode if args.component.endswith('encode') else component.decode
        inputs, kwargs = transfer(tuple(inputs)), transfer(kwargs)
        # CUDA calls the embedding argument inputs_embeds; shared TT callable
        # names it hidden. Preserve is_causal and input position exactly.
        if 'inputs_embeds' in kwargs:
            kwargs['hidden'] = kwargs.pop('inputs_embeds')
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        actual = component(*inputs, **kwargs)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        actual = ttnn.to_torch(actual)
        if codec:
            actual = actual.squeeze(1).transpose(1, 2).contiguous()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, args.output.with_suffix('.pt'))
        result = compare_tensors(expected, actual, min_pcc=args.min_pcc,
                                 max_relative_rms=args.max_relative_rms, max_abs=args.max_abs)
        report = {'mode': 'component_replay', 'component': args.component, 'event': event['key'],
                  'checkpoint': manifest['checkpoint'], 'dtype': args.dtype,
                  'device_id': args.device_id, 'arch': str(device.arch()),
                  'visible_devices': os.environ.get('TT_VISIBLE_DEVICES'),
                  'l1_small_size': getattr(args, 'l1_small_size', 262144),
                  'ttnn_version': importlib.metadata.version('ttnn'),
                  'torch_version': torch.__version__,
                  'math_fidelity': 'HiFi4', 'fp32_dest_acc': True,
                  'cold_launch_seconds': elapsed, 'metrics': result.to_dict(),
                  'limitations': ['Compilation may be included; not steady-state performance',
                                  'Captured CUDA input; not integrated TT generation',
                                  'MiniCPM prefill KV cache output is not qualified']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        logger.info('{}: {}', event['key'], result)
        return 0 if result.passed else 1
    finally:
        ttnn.close_device(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--component', choices=COMPONENTS, required=True)
    parser.add_argument('--event-index', type=int, default=0)
    parser.add_argument('--device-id', type=int, required=True)
    parser.add_argument('--l1-small-size', type=int, default=262144)
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--min-pcc', type=float, default=0.99)
    parser.add_argument('--max-relative-rms', type=float)
    parser.add_argument('--max-abs', type=float)
    parser.add_argument('--output', type=Path, required=True)
    return replay(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
