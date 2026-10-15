import argparse
import glob

from transformers import AutoTokenizer

from models.experimental.deepseek_v4_flash.demo.client import Client, OpenAIError, _stream_chat

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000")
# The server decodes greedily with no reply cap by default, so a repetition loop runs until the context is full.
parser.add_argument("--max-tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=None)
parser.add_argument("--top-p", type=float, default=None)
parser.add_argument("--think", action="store_true")
args = parser.parse_args()

snap = sorted(glob.glob("/home/tt-admin/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*"))[
    -1
]
tok = AutoTokenizer.from_pretrained(snap)
lines, total = [], 0
for line in open("/home/tt-admin/smanoj/top-200000.txt"):
    total += len(tok(line, add_special_tokens=False)["input_ids"])
    if total > 500_000:
        break
    lines.append(line)
print(f"sending {len(lines)} words, about {total} tokens")

client = Client(args.base_url, model="deepseek-v4.1-flash", timeout=3600)
params = {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p}
if args.think:
    params["thinking"] = True
print("commands: /reset clears the conversation, /exit quits; ctrl-C stops a reply")
# The server keeps the KV cache of the conversation, so each follow-up resends the full history and only the new
# turn is prefilled.
messages = []
prompt = "".join(lines) + '\nWhat word comes right after and before "' + lines[69000].strip() + '" in the list above?'

while True:
    messages.append({"role": "user", "content": prompt})
    try:
        reasoning, content = _stream_chat(client, messages, params)
    except OpenAIError as e:
        print(f"\n[error: {e}]")
        messages.pop()
    except KeyboardInterrupt:
        # The server stores the reply up to where it noticed the hang-up, which this side never sees, so the
        # history no longer matches and the next turn re-prefills the whole conversation.
        print("\n[reply stopped; the next turn re-prefills the conversation]")
        messages.pop()
    else:
        assistant = {"role": "assistant", "content": content}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)

    prompt = ""
    while not prompt:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if prompt in ("/exit", "/quit"):
            raise SystemExit(0)
        if prompt == "/reset":
            messages = []
            print("[conversation cleared]")
            prompt = ""
