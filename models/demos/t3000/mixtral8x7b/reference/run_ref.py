# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def main():
    # To avoid having a double copy of the weights, you can use the repacked weights instead
    use_repack_weights = True

    model_args = TtModelArgs()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    if use_repack_weights:  # Use repacked weights also used by TTNN model
        state_dict = model_args.load_state_dict()
    else:  # Use default HF weights
        state_dict = {}
        for i in range(1 + (n_layers - 1) // 4):
            state_dict_i = torch.load(model_args.consolidated_weights_path(i), map_location="cpu")
            state_dict.update(state_dict_i)

    # Encode prompts
    prompts = [""] * 32
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Load reference model
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = 20
    seqlen = 1  # Generating one token per user at a time
    batch = 32

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    # Keep track of generated outputs to print out later
    all_outputs_ref = []
    all_logits = []

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        # Run reference model
        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(pt_decode_input, positions).detach().float()
        all_logits.append(ref_output)

        pt_out_tok = torch.argmax(ref_output, dim=-1).squeeze(1)
        pt_decode_input = embd(pt_out_tok).view(batch, seqlen, -1)
        all_outputs_ref.append(pt_out_tok[0].tolist())

        # Print the generated output every iteration
        print("[User 0] Ref generation: ", "".join(tokenizer.decode(all_outputs_ref)))

    # Save output logits
    torch.save(all_logits, "ref_logits.pt")


if __name__ == "__main__":
    main()
