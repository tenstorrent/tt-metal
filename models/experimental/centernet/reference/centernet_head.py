import torch.nn as nn


class CTResNetHead(nn.Module):
    def __init__(self, parameters=None, init_cfg=None) -> None:
        super().__init__()
        self.parameters = parameters
        self.heatmap_head = self._build_head(80, self.parameters, "bbox_head.heatmap_head")
        self.wh_head = self._build_head(2, self.parameters, "bbox_head.wh_head")
        self.offset_head = self._build_head(2, self.parameters, "bbox_head.offset_head")

    def _build_head(self, out_channels: int, parameters, base_address) -> nn.Sequential:
        """Build head for each branch."""
        conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        conv1.weight = nn.Parameter(parameters[f"{base_address}.0.weight"])
        conv1.bias = nn.Parameter(parameters[f"{base_address}.0.bias"])

        conv2 = nn.Conv2d(64, out_channels, kernel_size=1)
        conv2.weight = nn.Parameter(parameters[f"{base_address}.2.weight"])
        conv2.bias = nn.Parameter(parameters[f"{base_address}.2.bias"])
        layer = nn.Sequential(conv1, nn.ReLU(inplace=True), conv2)
        return layer

    def forward(self, x):
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        wh_pred = self.wh_head(x)
        offset_pred = self.offset_head(x)

        return center_heatmap_pred, wh_pred, offset_pred
