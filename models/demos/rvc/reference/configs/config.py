import torch


class Config:
    def __init__(self):
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instead: str | None = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def params_config(self) -> tuple:
        # 5G GPU_RAM conf
        x_pad = 1
        x_query = 6
        x_center = 38
        x_max = 41
        return x_pad, x_query, x_center, x_max

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.params_config()

    def device_config(self) -> tuple:
        return self.params_config()
