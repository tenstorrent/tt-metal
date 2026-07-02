import torch
import ttnn
from transformers import AutoModelForCausalLM, AutoConfig
from models.demos.llasa.tt.model_def import TtLlasaModel, custom_preprocessor

def run_demo():
    model_id = "HKUST-Audio/Llasa-3B"
    
    print(f"Loading {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        torch_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        torch_model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Input (Text Prompt)
    # Llasa uses special tokens for text/audio. 
    # For basic verification, we can just feed random tokens or simple text if tokenizer works.
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    # Run PyTorch
    print("Running PyTorch model...")
    with torch.no_grad():
        torch_output = torch_model(input_ids)
        torch_logits = torch_output.logits

    # Run TTNN
    print("Initializing TTNN...")
    device = ttnn.open_device(device_id=0)
    
    try:
        print("Preprocessing weights...")
        parameters = ttnn.model_preprocessing.preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=custom_preprocessor,
            device=device
        )
        
        print("Initializing TtLlasaModel...")
        tt_model = TtLlasaModel(config, parameters, device)
        
        # Prepare Input
        tt_input_ids = ttnn.from_torch(
            input_ids, 
            dtype=ttnn.uint32, 
            layout=ttnn.ROW_MAJOR_LAYOUT, 
            device=device
        )
        
        print("Running TTNN model...")
        tt_logits = tt_model(tt_input_ids)
        
        # Comparison
        tt_logits_torch = ttnn.to_torch(tt_logits)
        
        print(f"PyTorch logits shape: {torch_logits.shape}")
        print(f"TTNN logits shape: {tt_logits_torch.shape}")
        
        # Compare
        # Note: Precision differences expected (BF16 vs FP16/FP32)
        # Check specific token correctness if possible, or correlation
        
        print("Demo finished.")
        
    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    run_demo()
