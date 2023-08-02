import torch





class PytorchEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, input_ids, token_type_ids=None):
        return self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

class TtEmbeddings(PytorchEmbeddings):
    def __init__(self, hugging_face_reference_model):
        super().__init__(hugging_face_reference_model)

    def forward(self, input_ids, token_type_ids=None):
        return super().forward(input_ids, token_type_ids)


def run_embeddings_inference():
    return
