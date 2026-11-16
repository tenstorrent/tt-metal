from pathlib import Path

s = Path("/tmp/build_correct_swe_actions.py").read_text()
a = s.index("calls=")
b = s.index(";jobs=[]", a)
s = s[:a] + "calls={'sphinx-doc__sphinx-10323':68,'pylint-dev__pylint-4661':19}" + s[b:]
s = s.replace("F=D/'anomalies/swe85-below-band/full-replies'", "F=D/'anomalies/swe85-below-band/full-replies-update10'")
insert = """
 full=F/f'{task}.call{n}.case.private.json';fc0=json.loads(full.read_text());tr0=json.loads(pathlib.Path(fc0['metadata']['native_trajectory']).read_text())
 if task.endswith('4661'):
  cmd=json.loads(tr0['messages'][36]['tool_calls'][0]['function']['arguments'])['command'];header="cat <<'EOF' > /testbed/pylint/config/__init__.py\\n";assert cmd.startswith(header);body=cmd[len(header):].split('\\nEOF')[0]+'\\n';(pre/'pylint/config/__init__.py').write_text(body);edits=[{'native_message':36,'file':'pylint/config/__init__.py','effect':'Exact quoted heredoc whole-file rewrite supersedes earlier debug sed30/34; no shell execution.'}]
 if task.endswith('10323'):
  p=pre/'sphinx/directives/code.py';lines=p.read_text().splitlines(True)
  # Native124 resets to image HEAD, then moves closing dedent line into index224;128 replaces complete223:229 region. No other source changes survive.
  start=next(i for i,line in enumerate(lines) if 'filters = [' in line);end=next(i for i in range(start,len(lines)) if 'self.dedent_filter' in lines[i]);line=lines.pop(end).strip();line+=',' if not line.endswith(',') else '';lines.insert(start+1,'                       '+line+'\\n')
  cmd=json.loads(tr0['messages'][128]['tool_calls'][0]['function']['arguments'])['command'];import shlex
  expression=shlex.split(cmd)[2];assert expression.startswith('223,229c\\\\');replacement=expression[len('223,229c\\\\'):].replace('\\\\n','\\n')+'\\n';lines[222:229]=replacement.splitlines(True);p.write_text(''.join(lines));edits=[{'native_message':124,'file':'sphinx/directives/code.py','effect':'Audited git checkout resets earlier debug and partial rewrites; literal move_filter_v2 list transformation reproduced.'},{'native_message':128,'file':'sphinx/directives/code.py','effect':'Exact sed223,229c source replacement decoded using shlex, no agent command execution.'}]
"""
s = s.replace(
    " shutil.copytree(pre,post,dirs_exist_ok=True);patch=",
    insert + " shutil.copytree(pre,post,dirs_exist_ok=True);patch=",
)
# Target Sphinx base+reference while state-relative diff preserves actual prior changes elsewhere.
s = s.replace(
    "dry=subprocess.run(['patch'",
    "\n if task.endswith('10323'):shutil.copyfile(base/'sphinx/directives/code.py',post/'sphinx/directives/code.py')\n dry=subprocess.run(['patch'",
)
s = s.replace(
    "for p in pre.rglob('*.py'):",
    "for p in pre.rglob('*'):\n  if not p.is_file() or '.git' in p.relative_to(pre).parts:continue",
)
s = s.replace(
    "(base/'.git/refs/heads/main').read_text().strip()",
    "next(p for p in (base/'.git/refs/heads').glob('*') if p.is_file()).read_text().strip()",
)
s = s.replace("O/'candidate-action-cases.json'", "O/'10323-4661-candidate-action-cases.json'").replace(
    "O/'candidate-actions-manifest.json'", "O/'10323-4661-candidate-actions-manifest.json'"
)
Path("/tmp/build_10323_4661.py").write_text(s)
