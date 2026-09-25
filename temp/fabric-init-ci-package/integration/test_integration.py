import unittest
import yaml
import apply_integration as p

BUILD = '''name: build
on:
  workflow_call:
    inputs:
      tracy:
        type: boolean
        default: false
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: build
        run: |
          build_args=(--build-metal-tests)
          ./build_metal.sh "${build_args[@]}"
'''
PARENT = '''name: parent
on:
  workflow_dispatch:
    inputs:
      run-fabric-perf-tests:
        type: boolean
        default: true
jobs:
  build-artifact:
    uses: ./.github/workflows/build-artifact.yaml
    with:
      tracy: true
  fabric-perf-tests:
    if: ${{ inputs.run-fabric-perf-tests != false }}
    uses: ./.github/workflows/tm-fabric-tests-perf-impl.yaml
    with:
      docker-image: image
'''
PERF = '''name: perf
on:
  workflow_call:
    inputs:
      docker-image:
        type: string
jobs:
  perf:
    runs-on: runner
    steps:
      - run: echo perf
'''
class IntegrationTests(unittest.TestCase):
    def test_build_with_new_environment(self):
        text=p.patch_build(BUILD)
        root=yaml.safe_load(text)
        step=root['jobs']['build']['steps'][0]
        self.assertEqual(step['env']['FABRIC_INIT_TRACY_CATEGORIES'],"${{ inputs.tracy-debug-categories || 'off' }}")
        self.assertIn('build_args+=(--build-perf-debug',step['run'])
        self.assertEqual(text,p.patch_build(text))
    def test_build_preserves_existing_environment(self):
        text=p.patch_build(BUILD.replace('        run: |','        env:\n          EXISTING: keep\n        run: |'))
        step=yaml.safe_load(text)['jobs']['build']['steps'][0]
        self.assertEqual(step['env']['EXISTING'],'keep')
        self.assertIn('FABRIC_INIT_TRACY_CATEGORIES',step['env'])
    def test_current_array_executable_style(self):
        source=BUILD.replace('build_args=(--build-metal-tests)', 'build_args=("./build_metal.sh" --build-metal-tests)').replace('./build_metal.sh "${build_args[@]}"', '"${build_args[@]}"')
        text=p.patch_build(source)
        self.assertIn('build_args+=(--build-perf-debug',text)
        self.assertEqual(text,p.patch_build(text))
    def test_artifact_suffix(self):
        source=BUILD + '      - run: |\n          TRACY_SUFFIX="_profiler"\n          echo "$TRACY_SUFFIX"\n'
        text=p.patch_build(source)
        self.assertIn("'_fabric-init'",text)
        yaml.compose(text)
    def test_parent_wiring(self):
        text=p.patch_parent(PARENT); data=yaml.safe_load(text)
        self.assertIn('tracy-debug-categories',data['jobs']['build-artifact']['with'])
        self.assertIn('run-fabric-init-perf-tests',data['jobs']['fabric-perf-tests']['if'])
        self.assertEqual(text,p.patch_parent(text))
    def test_perf_nested_workflow(self):
        text=p.patch_perf(PERF); data=yaml.safe_load(text)
        self.assertEqual(data['jobs']['fabric-init-tests']['uses'],'./.github/workflows/fabric-init-perf-impl.yaml')
        self.assertIn('perf',data['jobs'])
        self.assertEqual(text,p.patch_perf(text))
    def test_missing_anchor_refused(self):
        with self.assertRaises(ValueError):p.patch_build(BUILD.replace('./build_metal.sh','./other.sh'))
    def test_duplicate_anchor_refused(self):
        with self.assertRaises(ValueError):p.patch_build(BUILD.replace('          ./build_metal.sh', '          ./build_metal.sh "${build_args[@]}"\n          ./build_metal.sh'))
if __name__=='__main__':unittest.main()
