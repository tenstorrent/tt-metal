import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def function_subalpha(input, other, alpha):
    input = torch.as_tensor(input)
    other = torch.as_tensor(other)
    subalpha = torch.sub(input, other, alpha=alpha)
    return subalpha


def custom_subalpha(input, other, alpha):
    result = input + (-1.0 * alpha) * other
    return result


x = np.linspace(1, 100, 10)
y = np.linspace(1, 100, 10)
alpha = random.randint(1, 100)
z = function_subalpha(x, y, alpha)
z1 = custom_subalpha(x, y, alpha)
plt.plot(x, z, "--g", label="subalpha")
plt.plot(x, z1, "+r", label="custom subalpha")
plt.legend(loc="upper center")
plt.show()
