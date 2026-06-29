import torch
from diffusers import LongCatImagePipeline

MODULE_COMPONENTS = ("text_encoder", "transformer", "vae")


def print_module_tree(pipe):
    name_or_path = getattr(pipe.config, "_name_or_path", "meituan-longcat/LongCat-Image")
    print(f"LongCatImagePipeline ({name_or_path})\n")

    for name in MODULE_COMPONENTS:
        module = getattr(pipe, name)
        print("=" * 80)
        print(name)
        print("=" * 80)
        print(module)
        print()

    print("=" * 80)
    print("other pipeline components (not nn.Module trees)")
    print("=" * 80)
    print(f"scheduler: {pipe.scheduler.__class__.__name__}")
    print(f"tokenizer: {pipe.tokenizer.__class__.__name__}")
    print(f"text_processor: {pipe.text_processor.__class__.__name__}")


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        pipe = LongCatImagePipeline.from_pretrained("meituan-longcat/LongCat-Image", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    else:
        pipe = LongCatImagePipeline.from_pretrained("meituan-longcat/LongCat-Image", torch_dtype=torch.float32)
        pipe.to("cpu")

    print_module_tree(pipe)
