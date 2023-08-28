import numpy as np
import matplotlib.pyplot as plt
import torch

#log(2^x + 2^y)

def custom_implementation(in_x, in_y):
  #result = np.log2(np.power(2, in_x) + np.power(2, in_y));
  result = np.log2(np.exp(0.6931471805599453 * in_x) + np.exp(0.6931471805599453 * in_y))
  return result


x = np.linspace(-100, 100, 500)
y = np.linspace(-100, 100, 500)

t_x = torch.from_numpy(x)
t_y = torch.from_numpy(y)
t_out = torch.logaddexp2(t_x, t_y)
cust_out = custom_implementation(x, y);


plt.plot(y, t_out,'ob', label="torch result")
plt.plot(y, cust_out, '*r', label="custom result")
plt.legend(loc='upper center')
