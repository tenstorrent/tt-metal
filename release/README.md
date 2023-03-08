# Release

- Releasing
- Machine setup (for devs/customers)

# Releasing

rk: TODO

# Machine setup

## Requirements

In this directory, after doing a ``make build``:

```
source <REPO_HOME>/build/python_env/bin/activate
python -m pip install -r requirements-release.txt
sudo apt install sshpass
sudo apt install ssh-copy-id
```

Copy your SSH key to the machine(s) you want to setup.

```
ssh-copy-id <user>@<ip>
```

to every ``<ip>``.

**NOTE**: You could do something like this in order cut down on commands,
given that ``machines`` is a text file with each IP separated by a newline.

```
cat machines | xargs -n 1 -I{} ssh-copy-id <user>@{}
```

Then create an inventory file with ``ansible_user`` set for all IPs in the
``machine_setup`` directory.

Install the dependencies, hugepages, and check for the entries in the
inventory.

```
ansible-playbook -i machine_setup/inventory/<INV_FILE> machine_setup/playbooks/0-install-deps.yaml -K --ask-pass
ansible-playbook -i machine_setup/inventory/<INV_FILE> machine_setup/playbooks/1-install-hugepages.yaml -K --ask-pass
ansible-playbook -i machine_setup/inventory/<INV_FILE> machine_setup/playbooks/2-reboot.yaml -K --ask-pass
ansible-playbook -i machine_setup/inventory/<INV_FILE> machine_setup/playbooks/1-install-hugepages.yaml -K --ask-pass
```

**NOTE**: You can add ``--forks <number-of-forks>`` to speed up the Ansible
calls.
