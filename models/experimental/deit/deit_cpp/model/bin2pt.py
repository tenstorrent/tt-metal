import torch
from PIL import Image
from transformers import ViTFeatureExtractor, DeiTModel

# 定义模型路径
model_path = "/home/openkylin/.cache/huggingface/hub/models--facebook--deit-base-distilled-patch16-224/snapshots/155831199e645cc8ec9ace65a38ff782be6217e1"

# 1. 加载模型及其权重
print("正在加载 DeiT 模型...")
model = DeiTModel.from_pretrained(model_path, local_files_only=True)
model.eval()  # 切换到评估模式

# 2. 加载特征提取器
print("正在加载特征提取器...")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path, local_files_only=True)

# 3. 创建示例输入
print("正在创建示例输入...")
dummy_image = Image.new('RGB', (224, 224), color='red')
inputs = feature_extractor(images=dummy_image, return_tensors="pt", size=224)
example_input = inputs['pixel_values']

print(f"输入张量形状: {example_input.shape}")

# 创建一个包装器来返回张量而不是字典
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # 返回 last_hidden_state 而不是整个字典
        return outputs.last_hidden_state

# 使用包装器
wrapped_model = ModelWrapper(model)
wrapped_model.eval()

# 使用 torch.jit.trace 导出模型
print("开始进行 torch.jit.trace...")
traced_model = torch.jit.trace(wrapped_model, example_input)

# 将导出的模型保存到文件
output_filename = "deit_traced_model.pt"
traced_model.save(output_filename)

print(f"模型已成功导出为 {output_filename}")