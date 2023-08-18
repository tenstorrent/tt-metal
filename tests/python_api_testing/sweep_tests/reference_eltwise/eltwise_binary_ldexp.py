import torch
import numpy as np
import matplotlib.pyplot as plt


def function_ldexp(input, other):
    input = torch.as_tensor(input)
    other = torch.as_tensor(other)
    ldexp = torch.ldexp(input, other)
    return ldexp


def custom_ldexp(input, other):
    result = input * np.power(2, other)
    return result


x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
z = function_ldexp(x, y)
z1 = custom_ldexp(x, y)
plt.plot(x, z, "--g", label="ldexp")
plt.plot(x, z1, "+r", label="custom ldexp")
plt.legend(loc="upper center")
plt.show()
