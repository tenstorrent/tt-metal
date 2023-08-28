import torch
import numpy as np
import matplotlib.pyplot as plt


def atan(y):
    if abs(y) > 1:
        t3 = 1 / abs(y)
    else:
        t3 = abs(y)

    t4 = t3 * t3
    t0 = -float(0.013480470)
    t0 = t0 * t4 + float(0.057477314)
    t0 = t0 * t4 - float(0.121239071)
    t0 = t0 * t4 + float(0.195635925)
    t0 = t0 * t4 - float(0.332994597)
    t0 = t0 * t4 + float(0.999995630)
    t3 = t0 * t3

    if abs(y) > 1:
        t3 = 1.570796327 - t3
    else:
        t3 = t3

    if y < 0:
        t3 = -t3
    else:
        t3 = t3

    return t3


def atan2_impl(y, x):
    # Calculate the absolute values of x and y
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)

    raw_angle = atan(abs_y / abs_x)

    if x > 0:
        if y >= 0:
            angle = raw_angle
        else:
            angle = -raw_angle
    elif x < 0:
        if y > 0:
            angle = torch.pi - raw_angle
        elif y < 0:
            angle = -torch.pi + raw_angle
        else:
            angle = torch.pi
    elif x == 0:
        if y > 0:
            angle = torch.pi / 2
        elif y < 0:
            angle = -torch.pi / 2
        else:
            angle = 0.0

    return angle


# Create a sample input tensor
x = torch.tensor(np.linspace(-100, 100, 100))
y = torch.tensor(np.linspace(-100, 100, 100))

output = torch.empty_like(x)

result = torch.arctan2(y, x)

for i, (x_val, y_val) in enumerate(zip(x, y)):
    output[i] = atan2_impl(y_val, x_val)

# Plot the results
plt.plot(x, result, "ob", label="torch_atan2")
plt.plot(x, output, "+r", label="tt_atan2")

plt.legend()
plt.xlabel("x")
plt.ylabel("result")
plt.title("atan2(y/x) and tt atan2")
plt.show()
