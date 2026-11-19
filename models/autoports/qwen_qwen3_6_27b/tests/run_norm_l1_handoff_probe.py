"""Same-FP32-result norm BF16 conversion check; parent serializes device use.

This is an isolated conversion/geometry diagnostic, not a full-layer latency
claim. Real folded HF gamma and random compact activations exercise native
eight-core norm. The two paths consume the very same FP32 norm result.
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import ttnn

MODEL = 'Qwen/Qwen3.6-27B'
REVISION = '8f33b98937c6031bb49e713dee3e8e185aafd26f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--batch', type=int, nargs='+', default=[1,3,32])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(145145)
    checkpoint = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    config = json.loads((checkpoint/'config.json').read_text())
    config = config.get('text_config', config)
    assert config['layer_types'][args.layer] == 'full_attention'
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    gammas = {}
    for name in ('input_layernorm.weight','post_attention_layernorm.weight'):
        key = f'model.language_model.layers.{args.layer}.{name}'
        with safe_open(checkpoint/index[key], framework='pt', device='cpu') as f:
            gammas[name] = (1 + f.get_tensor(key).float()).reshape(1,1,5120)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,1), physical_device_ids=[0], trace_region_size=0)
    report = {'model':MODEL,'revision':REVISION,'layer':args.layer,'rows':[],
              'scope':'conversion geometry and own-input exactness; no performance claim'}
    try:
        memory = ttnn.create_sharded_memory_config(
            (32,640), ttnn.CoreGrid(x=8,y=1), ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8,1),subblock_w=4,block_h=1,block_w=20,inplace=False)
        for name, gamma in gammas.items():
            weight = ttnn.from_torch(gamma,device=mesh,dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for batch in args.batch:
                assert 1 <= batch <= 32
                for scale in (0.,.001,1.,32.):
                    x = (torch.randn(1,batch,5120)*scale).bfloat16()
                    token = ttnn.from_torch(x,device=mesh,dtype=ttnn.bfloat16,
                                           layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    xf = ttnn.interleaved_to_sharded(token,memory,output_dtype=ttnn.float32)
                    normalized = ttnn.rms_norm(xf,weight=weight,epsilon=config['rms_norm_eps'],
                                              compute_kernel_config=ck,program_config=program)
                    old_dram = ttnn.sharded_to_interleaved(normalized,ttnn.DRAM_MEMORY_CONFIG,
                                                         output_dtype=ttnn.bfloat16)
                    old = ttnn.to_memory_config(old_dram,memory)
                    new = ttnn.typecast(normalized,ttnn.bfloat16,memory_config=memory)
                    reused = ttnn.to_memory_config(new,memory)
                    assert reused.buffer_address() == new.buffer_address()
                    assert old.memory_config() == new.memory_config() == memory
                    read = lambda t: ttnn.to_torch(ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG),
                                                  mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0))
                    a,b = read(old),read(new)
                    row = {'norm':name,'batch':batch,'scale':scale,'bit_equal':torch.equal(a,b),
                           'mismatched_values':int((a!=b).sum()),
                           'max_abs':float((a.float()-b.float()).abs().max()),
                           'projection_conversion_alias':True}
                    report['rows'].append(row)
                    del token,xf,normalized,old_dram,old,new,reused
            del weight
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert report['rows'] and all(r['bit_equal'] for r in report['rows']), report
    print(json.dumps({'rows':len(report['rows']),'all_bit_equal':True,'output':str(args.output)}))


if __name__ == '__main__':
    main()
