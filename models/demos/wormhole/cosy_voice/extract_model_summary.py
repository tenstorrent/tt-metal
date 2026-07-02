import sys

sys.path.insert(0, "/root/tt-metal/models/demos/wormhole/cosy_voice/ref/CosyVoice")

import os

from hyperpyyaml import load_hyperpyyaml


def main():
    model_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
    hyper_yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")

    # Load config and initialize models
    with open(hyper_yaml_path, "r") as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")})

    flow = configs["flow"]
    hift = configs["hift"]

    with open("flow_summary.txt", "w") as f:
        f.write("============== FLOW MODEL ==============\n")
        f.write(str(flow))
        f.write("\n\n")

    with open("hift_summary.txt", "w") as f:
        f.write("============== HIFT MODEL ==============\n")
        f.write(str(hift))
        f.write("\n\n")

    print("Summaries written to flow_summary.txt and hift_summary.txt")


if __name__ == "__main__":
    main()
