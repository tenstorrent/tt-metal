import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_implementation(in_x, in_y):

  if (in_y == 0):
    result = (in_x != 0)
  else:
    r1 = (in_x != 0)
    r2 = (in_y != 0)
    add_r = r1 + r2
    result = add_r > 0

  return result


x = np.linspace(-100, 100, 60, dtype=np.float32)
y = 5

t_x = torch.from_numpy(x)
t_out = torch.logical_or(t_x, torch.tensor(y))

cust_out = custom_implementation(x, y)


plt.plot(x, t_out,'ob', label="torch result")
plt.plot(x, cust_out, '*r', label="custom result")
plt.legend(loc='upper center')
