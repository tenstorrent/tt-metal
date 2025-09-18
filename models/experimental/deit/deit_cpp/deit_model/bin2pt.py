import torch
from PIL import Image
from transformers import ViTFeatureExtractor, DeiTModel, DeiTForImageClassification, DeiTForImageClassificationWithTeacher

# 定义模型路径
model_path = "/home/openkylin/.cache/huggingface/hub/models--facebook--deit-base-distilled-patch16-224/snapshots/155831199e645cc8ec9ace65a38ff782be6217e1"

# 1. 加载两种模型及其权重
print("正在加载 DeiT 编码器模型...")
encoder_model = DeiTModel.from_pretrained(model_path, local_files_only=True)
encoder_model.eval()  # 切换到评估模式

print("正在加载 DeiT 分类模型...")
classifier_model = DeiTForImageClassification.from_pretrained(model_path, local_files_only=True)
classifier_model.eval()  # 切换到评估模式

print("正在加载 DeiT Teacher 分类模型...")
teacher_model = DeiTForImageClassificationWithTeacher.from_pretrained(model_path, local_files_only=True)
teacher_model.eval()  # 切换到评估模式

# 2. 加载特征提取器
print("正在加载特征提取器...")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path, local_files_only=True)

# 3. 创建示例输入
print("正在创建示例输入...")
dummy_image = Image.new('RGB', (224, 224), color='red')
inputs = feature_extractor(images=dummy_image, return_tensors="pt", size=224)
example_input = inputs['pixel_values']

print(f"输入张量形状: {example_input.shape}")

# 创建编码器包装器来返回 last_hidden_state
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # 返回 last_hidden_state 而不是整个字典
        return outputs.last_hidden_state

# 创建分类器包装器来返回 logits
class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # 返回分类 logits 而不是整个字典
        return outputs.logits

# 创建Teacher分类器包装器来返回三个输出
class TeacherWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # 返回 (logits, cls_logits, distillation_logits) 而不是整个字典
        return outputs.logits, outputs.cls_logits, outputs.distillation_logits

# 使用包装器
wrapped_encoder = EncoderWrapper(encoder_model)
wrapped_encoder.eval()

wrapped_classifier = ClassifierWrapper(classifier_model)
wrapped_classifier.eval()

wrapped_teacher = TeacherWrapper(teacher_model)
wrapped_teacher.eval()

# 使用 torch.jit.trace 导出编码器模型
print("开始进行编码器模型 torch.jit.trace...")
traced_encoder = torch.jit.trace(wrapped_encoder, example_input)

# 将编码器模型保存到文件
encoder_filename = "deit_encoder_model.pt"
traced_encoder.save(encoder_filename)
print(f"编码器模型已成功导出为 {encoder_filename}")

# 使用 torch.jit.trace 导出分类器模型
print("开始进行分类器模型 torch.jit.trace...")
traced_classifier = torch.jit.trace(wrapped_classifier, example_input)

# 将分类器模型保存到文件
classifier_filename = "deit_classifier_model.pt"
traced_classifier.save(classifier_filename)
print(f"分类器模型已成功导出为 {classifier_filename}")

# 使用 torch.jit.trace 导出Teacher分类器模型
print("开始进行Teacher分类器模型 torch.jit.trace...")
traced_teacher = torch.jit.trace(wrapped_teacher, example_input)

# 将Teacher分类器模型保存到文件
teacher_filename = "deit_teacher_model.pt"
traced_teacher.save(teacher_filename)
print(f"Teacher分类器模型已成功导出为 {teacher_filename}")

print("三个模型都已成功导出！")
print(f"- 编码器模型（包含特征提取）: {encoder_filename}")
print(f"- 分类器模型（包含 classifier.weight）: {classifier_filename}")
print(f"- Teacher分类器模型（包含双分类器和蒸馏功能）: {teacher_filename}")