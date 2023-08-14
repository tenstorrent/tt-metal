import numpy as np
import matplotlib.pyplot as plt
import torch

#log(e^x + e^y)

def custom_implementation(in_x, in_y):
  result = np.log(np.exp(in_x) + np.exp(in_y));
  return result


x = np.linspace(-90, 90, 500)
y = np.linspace(-90, 90, 500)

t_x = torch.from_numpy(x)
t_y = torch.from_numpy(y)
t_out = torch.logaddexp(t_x, t_y)
cust_out = custom_implementation(x, y)


plt.plot(y, t_out,'ob', label="torch result")
plt.plot(y, cust_out, '*r', label="custom result")
plt.legend(loc='upper center')
