import torch
import torch.nn as nn
from models.experimental.yolov4.reference import yolov4


def get_layer_details_to_csv(model, input_data, csv_filename="layer_details.csv"):
    layer_details = []

    def hook(module, input, output):
        layer_info = {
            "Layer Type": module.__class__.__name__,
            "Input Shape": list(input[0].size()),
            "Output Shape": list(output.size()),
            "Kernel Size": None,
            "Stride": None,
            "Padding": None,
            "Dilation": None,
            "Groups": None,
            "Weight Shape": None,
            "Bias Shape": None,
            "Num Features": None,
            "Activation": None,
        }
        # print(module.__class__.__name__)
        a = 0
        # Collect additional layer-specific attributes
        if isinstance(module, nn.Conv2d):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            layer_info["Kernel Size"] = module.kernel_size
            layer_info["Stride"] = module.stride
            layer_info["Padding"] = module.padding
            layer_info["Dilation"] = module.dilation
            layer_info["Groups"] = module.groups
            # print(layer_info['Input Shape'][0],", ",layer_info['Input Shape'][1],", ", layer_info['Output Shape'][1],", ", layer_info['Input Shape'][2],", ", layer_info['Input Shape'][3],", ", layer_info['Kernel Size'][0],", ",
            #       layer_info['Kernel Size'][1],", ", layer_info['Stride'][0],", ", layer_info['Stride'][1],", ", layer_info['Padding'][0],", ", layer_info['Padding'][1],", ", layer_info['Groups'],", ", "True",", ", "False")
        elif isinstance(module, nn.Linear):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Weight Shape"] = list(module.weight.size())
            layer_info["Bias Shape"] = list(module.bias.size()) if module.bias is not None else "None"
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
            # print(layer_info['Input Shape'],layer_info['Weight Shape'], layer_info['Bias Shape'], layer_info['in_features'], layer_info['out_features'])
        elif isinstance(module, nn.BatchNorm2d):
            layer_info["Num Features"] = module.num_features
            layer_info["Output Shape"] = list(output.size())
            # print(layer_info['Output Shape'])
        elif isinstance(module, nn.ReLU):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            layer_info["Activation"] = "ReLU"
        elif isinstance(module, nn.Identity):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
        elif isinstance(module, nn.Hardsigmoid):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
        elif isinstance(module, nn.MaxPool2d):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            layer_info["Kernel Size"] = module.kernel_size
            layer_info["Stride"] = module.stride
            layer_info["Padding"] = module.padding
            layer_info["Dilation"] = module.dilation
            layer_info["ceil_mode"] = module.ceil_mode
            print(
                layer_info["Input Shape"],
                " ,",
                layer_info["Kernel Size"],
                " ,",
                layer_info["Padding"],
                " ,",
                layer_info["Stride"],
                " ,",
                layer_info["ceil_mode"],
            )
        elif isinstance(module, nn.AvgPool2d):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
        elif isinstance(module, nn.Dropout):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            layer_info["p"] = module.p
            layer_info["inplace"] = module.inplace
        elif isinstance(module, nn.LayerNorm):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            # print(layer_info['Input Shape'][1]," ,", layer_info['Input Shape'][2])
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            layer_info["Input Shape"] = list(input[0].size())
            layer_info["Output Shape"] = list(output.size())
            # print(layer_info['Input Shape']," ,",layer_info['Input Shape'])
        # elif isinstance(module, nn.Mish):
        #     layer_info['Input Shape'] = list(input[0].size())
        #     layer_info['Output Shape'] = list(output.size())
        #     layer_info['Activation'] = 'Mish'
        #     print(layer_info['Input Shape'])
        elif isinstance(module, nn.Upsample):
            layer_info["scale_factor"] = module.scale_factor
            layer_info["input Shape"] = list(input[0].size())
            layer_info["mode"] = module.mode
            # print(layer_info['input Shape'], layer_info['scale_factor'], layer_info['mode'])

        layer_details.append(layer_info)

    # Register hooks for layers of interest
    hooks = []
    for layer in model.modules():
        if isinstance(
            layer,
            (
                nn.Conv2d,
                nn.Linear,
                nn.BatchNorm2d,
                nn.ReLU,
                nn.Identity,
                nn.Hardsigmoid,
                nn.MaxPool2d,
                nn.AvgPool2d,
                nn.Dropout,
                nn.Flatten,
                nn.SiLU,
                nn.LayerNorm,
                nn.GELU,
                nn.AdaptiveAvgPool1d,
                nn.Upsample,
            ),
        ):
            hooks.append(layer.register_forward_hook(hook))

    # Pass data through the model to trigger hooks
    model(input_data)

    # Remove hooks after use
    for h in hooks:
        h.remove()

    # Define fieldnames for the CSV
    fieldnames = [
        "Layer Type",
        "Input Shape",
        "Output Shape",
        "Kernel Size",
        "Stride",
        "Padding",
        "Dilation",
        "Groups",
        "Weight Shape",
        "Bias Shape",
        "Num Features",
        "Activation",
    ]

    # Write the layer details to a CSV file
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for layer_info in layer_details:
    #         writer.writerow(layer_info)

    print(f"Layer details saved to {csv_filename}")


# Example usage based on your provided code
input_data = torch.randn(1, 3, 640, 640)


model = yolov4.Yolov4()

with torch.no_grad():
    outputs = model(input_data)

model.eval()
reference_model = model
# for layer in reference_model.children():
#         print(layer)

# Load state dict as per your example
new_state_dict = {}
keys = [name for name, parameter in reference_model.state_dict().items()]
ds_state_dict = {k: v for k, v in reference_model.state_dict().items()}
values = [parameter for name, parameter in ds_state_dict.items()]
for i in range(len(keys)):
    new_state_dict[keys[i]] = values[i]

reference_model.load_state_dict(new_state_dict)
reference_model.eval()


# Get output (inference)
output = reference_model(input_data)

# Capture layer details and save to CSV
get_layer_details_to_csv(reference_model, input_data, csv_filename="vovnet_details.csv")
