import json, glob
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("/home/ubuntu/models/olmo31-32b-think")
f = glob.glob(
    "/home/ubuntu/ssinghal/olmo3/tt-inference-server/workflow_logs/reports_output/evals/*/*/*/*/samples_*.jsonl"
)[0]
d = [json.loads(l) for l in open(f)][1]
p = tk(d["arguments"]["gen_args_0"]["arg_0"], add_special_tokens=False).input_ids
ids = tk(d["resps"][0][0], add_special_tokens=False).input_ids
json.dump(
    {"prompt_len": len(p), "ids": p + ids[:24450]}, open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json", "w")
)
print(len(p))
