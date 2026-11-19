"""Real-weight host BFP8 packing probe for numerical-only zero-Z correction.

No device is opened. This tests packed stored elements, not decoder accuracy
or latency. Full short/512-step runs use the separately frozen decoder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import ttnn

MODEL='Qwen/Qwen3.6-27B'
REVISION='8f33b98937c6031bb49e713dee3e8e185aafd26f'


def fingerprint(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    checkpoint=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    pieces=[]
    for name in ('qkv','z','b','a'):
        key=f'model.language_model.layers.{args.layer}.linear_attn.in_proj_{name}.weight'
        with safe_open(checkpoint/index[key],framework='pt',device='cpu') as handle:
            value=handle.get_tensor(key)
        if name in ('b','a'):
            value=torch.nn.functional.pad(value,(0,0,0,16))
        pieces.append(value)
    original=torch.cat(pieces,dim=0).T.contiguous().float()
    del pieces,value
    assert tuple(original.shape)==(5120,16512)
    pack=lambda value:ttnn.from_torch(value,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT)
    primary=pack(original)
    quantized=ttnn.to_torch(primary).float()
    primary_hash=fingerprint(quantized)
    residual=original-quantized
    del original
    baseline=ttnn.to_torch(pack(residual)).float()
    residual[:,10240:16384]=0
    candidate=ttnn.to_torch(pack(residual)).float()
    del residual
    rows=[]
    for name,start,end in (('QKV',0,10240),('Z',10240,16384),('B_A_with_padding',16384,16512)):
        old,new=baseline[:,start:end],candidate[:,start:end]
        row={'region':name,'start':start,'end':end,'stored_elements_equal':torch.equal(old,new),
             'candidate_nonzero_count':int(torch.count_nonzero(new)),
             'baseline_nonzero_count':int(torch.count_nonzero(old)),
             'changed_elements':int((old!=new).sum())}
        rows.append(row)
    # Re-read the original packed primary object after all correction work.
    primary_unchanged=fingerprint(ttnn.to_torch(primary).float())==primary_hash
    report={'model':MODEL,'revision':REVISION,'layer':args.layer,'device_opened':False,
            'logical_shape':[5120,16512],'physical_N_policy':'unchanged; same decoder geometry',
            'primary_stored_values_unchanged':primary_unchanged,'primary_sha256':primary_hash,
            'rows':rows,'scope':'host real-weight BFP8 packing only; no decoder PCC or speed claim'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert primary_unchanged
    assert all(r['stored_elements_equal'] for r in rows if r['region']!='Z')
    assert next(r for r in rows if r['region']=='Z')['candidate_nonzero_count']==0
    print(json.dumps({'output':str(args.output),'all_checks_passed':True}))


if __name__=='__main__':
    main()
