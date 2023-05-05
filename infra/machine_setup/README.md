# Machine setup

## Requirements

In this directory, after doing a ``make build`` and using the ``infra`` env:

```
source <REPO_HOME>/build/python_env/bin/activate
python -m pip install -r ../requirements-infra.txt
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

## Machine setup

Create an inventory file with the following entries set for all IPs in the
``inventories`` directory.

* ``ansible_user``
* ``arch`` (ex. ``gs``, ``wh`` etc.)

Then, run the Ansible playbooks:

```
ansible-playbook -i inventory/<INV_FILE> playbooks/0-install-deps.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/3-install-drivers.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/2-reboot.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/4-install-flash.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/2-reboot.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/1-install-hugepages-part1.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/2-reboot.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/1-install-hugepages-part2.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/2-reboot.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/1-install-hugepages-verify.yaml
ansible-playbook -i inventory/<INV_FILE> playbooks/5-verify-hw.yaml
```

**NOTE**: You can add ``--forks <number-of-forks>`` to speed up the Ansible
calls.

**NOTE**: You can use the options ``-K``, ``--ask-pass``, and ``--extra-vars
'ansible_become_pass=<ansible become password>'`` as part of your options to
help with authentication.
