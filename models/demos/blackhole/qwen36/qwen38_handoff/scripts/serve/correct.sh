#!/bin/bash
D=$1; U=http://127.0.0.1:8000/v1/chat/completions
req(){ python3 - "$1" <<'PY'
import json,sys
print(json.dumps({"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":96,"chat_template_kwargs":{"enable_thinking":False}}))
PY
}
req "What is the capital of France? Answer in one sentence." > $D/a_req.json
curl -s $U -H 'Content-Type: application/json' -d @$D/a_req.json > $D/a_resp.json
i=0
for p in "Name three primary colors." "What is 12 times 12?" "Write one sentence about the ocean." "Translate 'good morning' into Spanish."; do
 i=$((i+1)); req "$p" > $D/b${i}_req.json
 curl -s $U -H 'Content-Type: application/json' -d @$D/b${i}_req.json > $D/b${i}_resp.json &
done; wait
for f in $D/a_resp.json $D/b?_resp.json; do echo "== $f"; python3 -c "import json,sys;d=json.load(open('$f'));print(d['choices'][0]['message']['content'] if 'choices' in d else d)"; done
