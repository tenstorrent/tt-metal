# SPDX-License-Identifier: Apache-2.0
"""A long multi-turn history with a short factual append and an awkward tail."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
root=Path('models/autoports/qwen_qwen3_8_27b')
tok=AutoTokenizer.from_pretrained('.cache/qwen38-full')
facts={'project':'Cedar','port':7319,'owner':'Mira'}
base=[{'role':'user','content':'Remember this deployment record for later: project Cedar, port 7319, owner Mira.'},
      {'role':'assistant','content':'Recorded: project Cedar, port 7319, owner Mira.'}]
words=tok.encode('Observation: the agent checked the build output and continued its investigation. ',add_special_tokens=False)
target=32769
n=target-150
for _ in range(20):
    filler=tok.decode((words*((n+len(words)-1)//len(words)))[:n])
    messages=base+[{'role':'user','content':'Here is the subsequent investigation log:\n'+filler},
                   {'role':'assistant','content':'I have read the investigation log.'},
                   {'role':'user','content':'Return only a JSON object with keys project, port, owner from the original deployment record. No explanation.'}]
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    if len(ids)==target:break
    n+=target-len(ids)
else:raise RuntimeError('Could not render exact workload')
case=dict(name='agent_history_32769',messages=messages,enable_thinking=False,input_tokens=ids,target_tokens=target,
          input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),expected_facts=facts)
path=root/'doc/datatype_sweep/agent_long_cases.json'
path.write_text(json.dumps(dict(chat_template_sha256=hashlib.sha256(tok.get_chat_template().encode()).hexdigest(),
    source='Synthetic multi-turn deployment history with short append, original facts near the beginning; not a performance benchmark',cases=[case]),indent=2)+'\n')
print(path,len(ids))
