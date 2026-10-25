# SPDX-License-Identifier: Apache-2.0
"""CPU HF logits at the exact first divergent decision; diagnostic only."""
import hashlib
import json
from pathlib import Path
import torch
from .hf_text import load_model

root=Path('models/autoports/qwen_qwen3_8_27b/doc/full_model')
reference=root/'benchmark_hf_control_512.json'
data=json.loads(reference.read_text())
torch.set_num_threads(16)
model,tokenizer=load_model()
cache=None
with torch.inference_mode():
    for step in range(57):
        inp=data['input_tokens'] if step==0 else [data['output_tokens'][step-1]]
        result=model(input_ids=torch.tensor([inp]),past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=result.past_key_values
        if step%16==0:print('HF_SHARED_HISTORY',step,flush=True)
    logits=result.logits[0,-1].float()
    scores,ids=logits.topk(100)
    record=dict(scope='HF BF16 exact common-history decision; CPU diagnostic, no performance claim',
        reference=str(reference),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(json.dumps(data['input_tokens']).encode()).hexdigest(),
        output_index=56,input_position=4151,history_tokens=56,selected_token=int(logits.argmax()),
        compared_tokens={str(t):dict(logit=float(logits[t]),rank=1+int((logits>logits[t]).sum())) for t in (1,279)},
        top100=[dict(token=int(t),logit=float(v)) for t,v in zip(ids,scores)],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=str(torch.__version__))
    assert record['selected_token']==data['output_tokens'][56]
    (root/'hf_long_quality_decision.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['compared_tokens'],flush=True)
