import torch
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def test_torch_prefill(device):
    seq_len = 8192 * 2
    batch = 1
    prompt_file = "models/demos/t3000/mixtral8x7b/demo/tale-of-two-cities.txt"
    with open(prompt_file, "r") as f:
        prompt = f.read()

    model_args = TtModelArgs(device)
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    encoded_prompts = tokenizer.encode(prompt)[:seq_len]
    encoded_prompts_tensor = torch.tensor(encoded_prompts)
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)

    attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()
    positions = torch.LongTensor(range(seq_len))
    ref_output = reference_model(pt_decode_input, positions, attn_mask_torch, mode="prefill").detach().float()
    torch.save(ref_output, "ref_output_prefil_16L_16k.pt")
