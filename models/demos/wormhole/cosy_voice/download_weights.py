import os

from huggingface_hub import snapshot_download


def main():
    target_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading FunAudioLLM/Fun-CosyVoice3-0.5B-2512 to {target_dir}...")
    snapshot_download(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=target_dir,
        ignore_patterns=["*.onnx", "*.pb"],  # we don't need ONNX models for TTNN bringup
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
