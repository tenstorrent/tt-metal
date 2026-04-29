import yaml
from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/tinyllama-tokenizer.json")
text = open("data/shakespeare.txt", "r", encoding="utf-8").read()
ids = tok.encode(text).ids
out = "data/tokenized_shakespeare_tinyllama.yaml"
with open(out, "w") as f:
    yaml.dump(
        {"tokenizer_vocab_size": tok.get_vocab_size(), "data_length": len(ids), "tokens": ids},
        f,
        default_flow_style=True,
    )
print("wrote", out, "tokens:", len(ids), "vocab_size:", tok.get_vocab_size())
