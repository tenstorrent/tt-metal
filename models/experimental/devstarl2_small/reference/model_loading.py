import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_name = "mistralai/Devstral-Small-2-24B-Instruct-2512"

# Load processor (handles both image + text)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Load model (Mistral3 maps via AutoModelForImageTextToText → Mistral3ForConditionalGeneration)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

_device = next(model.parameters()).device

# Load image
image = Image.open("models/experimental/devstarl2_small/reference/sample.jpeg").convert("RGB")

# Build prompt with the correct image placeholder(s). Mistral3 uses token id `image_token_id`
# (decoded as "[IMG]"), NOT the string "<image>". The processor expands one logical "[IMG]"
# in the chat template into one token per vision patch when `images=` is passed.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

# Move tensor inputs to the same device as the model (works with device_map="auto")
inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Generate output (avoid passing both max_length from config and max_new_tokens)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# Decode output
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print(output_text)
