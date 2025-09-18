import torch
from PIL import Image
from transformers import ViTFeatureExtractor, DeiTModel, DeiTForImageClassification, DeiTForImageClassificationWithTeacher

# Define model name (download from Hugging Face Hub online)
model_name = "facebook/deit-base-distilled-patch16-224"

# 1. Load models and their weights
print("Loading DeiT encoder model from Hugging Face Hub...")
encoder_model = DeiTModel.from_pretrained(model_name)
encoder_model.eval()  # Switch to evaluation mode

print("Loading DeiT classification model from Hugging Face Hub...")
classifier_model = DeiTForImageClassification.from_pretrained(model_name)
classifier_model.eval()  # Switch to evaluation mode

print("Loading DeiT Teacher classification model from Hugging Face Hub...")
teacher_model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)
teacher_model.eval()  # Switch to evaluation mode

# 2. Load feature extractor
print("Loading feature extractor from Hugging Face Hub...")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# 3. Create example input
print("Creating example input...")
dummy_image = Image.new('RGB', (224, 224), color='red')
inputs = feature_extractor(images=dummy_image, return_tensors="pt", size=224)
example_input = inputs['pixel_values']

print(f"Input tensor shape: {example_input.shape}")

# Create encoder wrapper to return last_hidden_state
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # Return last_hidden_state instead of the entire dictionary
        return outputs.last_hidden_state

# Create classifier wrapper to return logits
class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # Return classification logits instead of the entire dictionary
        return outputs.logits

# Create Teacher classifier wrapper to return three outputs
class TeacherWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # Return (logits, cls_logits, distillation_logits) instead of the entire dictionary
        return outputs.logits, outputs.cls_logits, outputs.distillation_logits

# Use wrappers
wrapped_encoder = EncoderWrapper(encoder_model)
wrapped_encoder.eval()

wrapped_classifier = ClassifierWrapper(classifier_model)
wrapped_classifier.eval()

wrapped_teacher = TeacherWrapper(teacher_model)
wrapped_teacher.eval()

# Export encoder model using torch.jit.trace
print("Starting torch.jit.trace for encoder model...")
traced_encoder = torch.jit.trace(wrapped_encoder, example_input)

# Save encoder model to file
encoder_filename = "deit_encoder_model.pt"
traced_encoder.save(encoder_filename)
print(f"Encoder model successfully exported as {encoder_filename}")

# Export classifier model using torch.jit.trace
print("Starting torch.jit.trace for classifier model...")
traced_classifier = torch.jit.trace(wrapped_classifier, example_input)

# Save classifier model to file
classifier_filename = "deit_classifier_model.pt"
traced_classifier.save(classifier_filename)
print(f"Classifier model successfully exported as {classifier_filename}")

# Export Teacher classifier model using torch.jit.trace
print("Starting torch.jit.trace for Teacher classifier model...")
traced_teacher = torch.jit.trace(wrapped_teacher, example_input)

# Save Teacher classifier model to file
teacher_filename = "deit_teacher_model.pt"
traced_teacher.save(teacher_filename)
print(f"Teacher classifier model successfully exported as {teacher_filename}")

print("All three models have been successfully exported!")
print(f"- Encoder model (with feature extraction): {encoder_filename}")
print(f"- Classifier model (with classifier.weight): {classifier_filename}")
print(f"- Teacher classifier model (with dual classifiers and distillation): {teacher_filename}")