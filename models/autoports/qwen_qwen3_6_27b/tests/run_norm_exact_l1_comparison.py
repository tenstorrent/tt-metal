"""Compare exact-tree norm with L1scratch to its DRAM incumbent using real gamma.

Device test, to be run only by the serialized parent. This does not itself prove
whole-decoder accuracy. Host Torch and safetensors are confined to this test.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
import ttnn

MODEL = 'Qwen/Qwen3.6-27B'
REVISION = '8f33b98937c6031bb49e713dee3e8e185aafd26f'
CANDIDATES = Path(__file__).parents[1] / 'doc/optimized_decoder/candidates'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def real_gammas(root, layers):
    from safetensors import safe_open
    if root is None:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    for layer in layers:
        for norm in ('input_layernorm', 'post_attention_layernorm'):
            name = f'model.language_model.layers.{layer}.{norm}.weight'
            with safe_open(root / mapping[name], framework='pt', device='cpu') as f:
                weight = f.get_tensor(name)
            assert weight.numel() == 5120
            yield name, (1 + weight.float()).reshape(1, 1, 5120)


def inputs(batch, generator, rounds):
    for iteration in range(rounds):
        scale = (.001, 1., 16., 1e-6)[iteration % 4]
        yield f'random_{iteration}_scale_{scale}', (torch.randn((1, batch, 5120), generator=generator) * scale).bfloat16()
    base = torch.arange(5120).reshape(1, 1, 5120).expand(1, batch, 5120)
    yield 'zero', torch.zeros((1, batch, 5120), dtype=torch.bfloat16)
    signed_zero = torch.zeros((1, batch, 5120), dtype=torch.bfloat16)
    signed_zero[..., ::2] = -0.0
    yield 'signed_zero', signed_zero
    yield 'powers_and_signs', (torch.pow(-1., base % 2) * torch.exp2((base % 25 - 12).float())).bfloat16()
    # One nonzero value in each of the five chunks, with different rows/columns.
    sparse = torch.zeros((1, batch, 5120), dtype=torch.bfloat16)
    for user in range(batch):
        for chunk in range(5):
            sparse[0, user, chunk * 1024 + ((user * 31 + chunk * 17) % 1024)] = 2. ** (chunk - 2)
    yield 'sparse_chunk_boundaries', sparse
    # Distinct mantissas and exponent bands stress the FP32 five-chunk fold.
    mantissa = 1 + (base % 128).float() / 128
    yield 'mixed_mantissas', (mantissa * torch.exp2((base // 1024 - 2).float()) * torch.pow(-1., base % 3)).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, default=CANDIDATES / 'hidden_rmsnorm_exact_tree_l1_80.py')
    ap.add_argument('--reference', type=Path, default=CANDIDATES / 'hidden_rmsnorm_distributed_exact_tree.py')
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--layers', type=int, nargs='+', default=[0])
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 3, 32])
    ap.add_argument('--stress-rounds', type=int, default=4)
    ap.add_argument('--real-tensors', type=Path, action='append', default=[],
                    help='Optional saved dict with x BF16 [1,B,5120] and folded gamma FP32 [1,1,5120].')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(9431)
    candidate = load_module(args.candidate, 'exact_candidate')
    reference = load_module(args.reference, 'exact_reference')
    gammas = list(real_gammas(args.model_dir, args.layers))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    saved_failure = False
    persistent = {}
    try:
        def up(tensor, dtype):
            return ttnn.from_torch(tensor, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        def compare(name, gamma_name, x, gamma):
            nonlocal saved_failure
            x = x.bfloat16().contiguous()
            gamma = gamma.float().contiguous()
            tx, tg = up(x, ttnn.bfloat16), up(gamma, ttnn.float32)
            batch = x.shape[1]
            if batch not in persistent:
                def empty(shape, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
                    return ttnn.empty(shape, dtype=dtype, device=mesh, layout=ttnn.TILE_LAYOUT,
                                      memory_config=memory)
                persistent[batch] = (empty(tuple(x.shape), ttnn.bfloat16), empty(tuple(x.shape), ttnn.bfloat16),
                    (empty((1, 32, 1024), ttnn.float32, ttnn.L1_MEMORY_CONFIG), empty((1, 32, 32), ttnn.float32, ttnn.L1_MEMORY_CONFIG)))
            ref_buffer, actual_buffer, scratch = persistent[batch]
            expected_t = reference.hidden_rmsnorm_sfpu(tx, tg, output=ref_buffer)
            actual_t = candidate.hidden_rmsnorm_sfpu(tx, tg, output=actual_buffer, scratch=scratch)
            read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).contiguous()
            expected, actual = read(expected_t), read(actual_t)
            mismatch = actual.view(torch.int16) != expected.view(torch.int16)
            relative = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20))
            row = dict(batch=batch, input=name, gamma=gamma_name, mismatches=int(mismatch.sum()),
                       elements=actual.numel(), relative_rms=relative)
            rows.append(row)
            print(row, flush=True)
            if row['mismatches'] and not saved_failure:
                torch.save(dict(x=x, gamma=gamma, expected=expected, actual=actual), args.output.with_suffix('.failure.pt'))
                saved_failure = True
            # Save partial evidence before subsequent cases or a device failure.
            args.output.write_text(json.dumps(dict(candidate=str(args.candidate), reference=str(args.reference),
                require_bit_exact=True, cases=rows), indent=2) + '\n')

        for batch in args.batches:
            for name, x in inputs(batch, generator, args.stress_rounds):
                for gamma_name, gamma in gammas:
                    compare(name, gamma_name, x, gamma)
        for path in args.real_tensors:
            sample = torch.load(path, map_location='cpu', weights_only=True)
            compare(str(path), 'captured_folded_gamma', sample['x'], sample['gamma'])
        assert all(row['mismatches'] == 0 for row in rows), 'Candidate must match incumbent BF16 bits in every case'
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
