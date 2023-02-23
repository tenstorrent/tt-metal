import torch

class TtEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

class PytorchEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, x):
        return self.embeddings(x)

def run_embeddings_inference():
    return

if __name__ == "__main__":
    # Initialize the device
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_embeddings_inference()
    _C.device.CloseDevice(device)
