import torch


class SpeculativeDecode(torch.nn.Module):
    def __init__(self, base_model, draft_model):
        super().__init__()
        self.base_model = base_model
        self.draft_model = draft_model

    def initialize_tree(self):
        pass

    def tree_decoding(self):
        pass

    def evaluate_posterior(self):
        pass

    def update_inference_inputs(self):
        pass
